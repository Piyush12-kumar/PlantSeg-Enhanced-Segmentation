# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom Activation Fusion for SAN
================================
Implements the exact fusion formula:

    A = Softmax((QK^T / sqrt(d)) * tau + beta * D)
    D = |F_SAN - F_CLIP|
    O = A * V
    G = sigmoid(W_g * F_CLIP)
    F_out = F_SAN + alpha * (G ⊙ O)

Where:
    - tau (τ): learnable temperature parameter
    - beta (β): learnable weight for difference term
    - alpha (α): learnable scaling for gated output
    - W_g: projection matrix for gating
    - ⊙: element-wise multiplication
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import trunc_normal_

from mmseg.registry import MODELS
from ..utils import LayerNorm2d, resize
from .san_head import (MLPMaskDecoder, RecWithAttnbias,
                       SideAdapterCLIPHead, SideAdapterNetwork)


class CustomActivationFusionSAN(SideAdapterNetwork):
    """SideAdapterNetwork with Custom Activation Fusion.
    
    Implements the fusion formula:
        A = Softmax((QK^T / sqrt(d)) * tau + beta * D)
        D = |F_SAN - F_CLIP|
        O = A * V
        G = sigmoid(W_g * F_CLIP)
        F_out = F_SAN + alpha * (G ⊙ O)
    
    Args:
        init_tau (float): Initial value for temperature tau. Default: 1.0.
        init_beta (float): Initial value for difference weight beta. Default: 0.1.
        init_alpha (float): Initial value for output scaling alpha. Default: 0.5.
        attn_heads (int): Number of attention heads. Default: 4.
        **kwargs: Additional arguments passed to SideAdapterNetwork.
    """

    def __init__(
        self,
        init_tau: float = 1.0,
        init_beta: float = 0.1,
        init_alpha: float = 0.5,
        attn_heads: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        embed_dims = kwargs.get('embed_dims', 240)
        self.attn_heads = attn_heads
        self.head_dim = embed_dims // attn_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d)
        
        # ================================================================
        # Learnable Parameters (per fusion point)
        # ================================================================
        num_fusion = len(self.fusion_index)
        
        # τ (tau): Temperature parameter for attention scaling
        self.tau = nn.ParameterList([
            nn.Parameter(torch.tensor([init_tau]))
            for _ in range(num_fusion)
        ])
        
        # β (beta): Weight for the difference term D
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor([init_beta]))
            for _ in range(num_fusion)
        ])
        
        # α (alpha): Scaling factor for gated output
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor([init_alpha]))
            for _ in range(num_fusion)
        ])
        
        # ================================================================
        # Attention Projections (Q, K, V)
        # ================================================================
        self.q_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims)
            for _ in range(num_fusion)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims)
            for _ in range(num_fusion)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims)
            for _ in range(num_fusion)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims)
            for _ in range(num_fusion)
        ])
        
        # ================================================================
        # Gating Network: G = σ(W_g * F_CLIP)
        # ================================================================
        self.gate_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims)
            for _ in range(num_fusion)
        ])
        
        # ================================================================
        # Layer Normalization
        # ================================================================
        self.norm_san = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_fusion)
        ])
        self.norm_clip = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_fusion)
        ])
        self.norm_out = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_fusion)
        ])

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        
        for i in range(len(self.fusion_index)):
            # Initialize projections with Xavier
            for proj in [self.q_proj[i], self.k_proj[i], 
                        self.v_proj[i], self.out_proj[i], 
                        self.gate_proj[i]]:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
            
            # Initialize layer norms
            for norm in [self.norm_san[i], self.norm_clip[i], self.norm_out[i]]:
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)

    def fuse_clip(
        self,
        fused_index: int,
        x: torch.Tensor,
        clip_feature: torch.Tensor,
        hwshape: Tuple[int, int],
        L: int
    ) -> torch.Tensor:
        """Custom Activation Fusion.
        
        Implements:
            A = Softmax((QK^T / sqrt(d)) * tau + beta * D)
            D = |F_SAN - F_CLIP|
            O = A * V
            G = sigmoid(W_g * F_CLIP)
            F_out = F_SAN + alpha * (G ⊙ O)
        
        Args:
            fused_index: Index of current fusion layer.
            x: Visual tokens [B, N+L, C] where N is num_queries.
            clip_feature: CLIP features [B, C_clip, H_clip, W_clip].
            hwshape: Target spatial shape (H, W).
            L: Number of spatial tokens (H * W).
            
        Returns:
            Fused features with same shape as input x.
        """
        B = x.shape[0]
        C = x.shape[-1]
        
        # ================================================================
        # Step 1: Prepare F_SAN and F_CLIP
        # ================================================================
        # F_SAN: Spatial tokens from the side adapter
        F_SAN = x[:, -L:, ...]  # [B, L, C]
        
        # F_CLIP: Project and reshape CLIP features
        fused_clip = self.conv_clips[fused_index](clip_feature.contiguous())
        fused_clip = resize(fused_clip, size=hwshape,
                           mode='bilinear', align_corners=False)
        # [B, C, H, W] -> [B, L, C]
        F_CLIP = fused_clip.permute(0, 2, 3, 1).reshape(B, L, C)
        
        # Normalize
        F_SAN_norm = self.norm_san[fused_index](F_SAN)
        F_CLIP_norm = self.norm_clip[fused_index](F_CLIP)
        
        # ================================================================
        # Step 2: Compute D = |F_SAN - F_CLIP|
        # ================================================================
        D = torch.abs(F_SAN_norm - F_CLIP_norm)  # [B, L, C]
        
        # ================================================================
        # Step 3: Compute Q, K, V projections
        # ================================================================
        Q = self.q_proj[fused_index](F_SAN_norm)   # [B, L, C]
        K = self.k_proj[fused_index](F_CLIP_norm)  # [B, L, C]
        V = self.v_proj[fused_index](F_CLIP_norm)  # [B, L, C]
        
        # Reshape for multi-head attention
        # [B, L, C] -> [B, num_heads, L, head_dim]
        Q = Q.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        
        # ================================================================
        # Step 4: Compute A = Softmax((QK^T / sqrt(d)) * tau + beta * D)
        # ================================================================
        # Standard attention scores: QK^T / sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [B, num_heads, L, L]
        
        # Get tau and beta
        tau = self.tau[fused_index]    # scalar
        beta = self.beta[fused_index]  # scalar
        
        # Apply temperature: (QK^T / sqrt(d)) * tau
        attn_scores = attn_scores * tau
        
        # Add difference bias: + beta * D
        # D is [B, L, C], we need to aggregate to [B, L, L] or broadcast
        # Using sum over channel dimension for the attention bias
        # D_bias: compute pairwise difference magnitude
        D_expanded = D.unsqueeze(2)  # [B, L, 1, C]
        D_bias = torch.sum(D_expanded, dim=-1)  # [B, L, 1] - per-position bias
        D_bias = D_bias.unsqueeze(1)  # [B, 1, L, 1] for broadcasting
        
        attn_scores = attn_scores + beta * D_bias
        
        # Apply softmax: A = Softmax(...)
        A = F.softmax(attn_scores, dim=-1)  # [B, num_heads, L, L]
        
        # ================================================================
        # Step 5: Compute O = A * V
        # ================================================================
        O = torch.matmul(A, V)  # [B, num_heads, L, head_dim]
        O = O.transpose(1, 2).reshape(B, L, C)  # [B, L, C]
        O = self.out_proj[fused_index](O)
        
        # ================================================================
        # Step 6: Compute G = sigmoid(W_g * F_CLIP)
        # ================================================================
        G = torch.sigmoid(self.gate_proj[fused_index](F_CLIP_norm))  # [B, L, C]
        
        # ================================================================
        # Step 7: Compute F_out = F_SAN + alpha * (G ⊙ O)
        # ================================================================
        alpha = self.alpha[fused_index]
        
        # G ⊙ O: element-wise multiplication
        gated_output = G * O  # [B, L, C]
        
        # F_out = F_SAN + alpha * (G ⊙ O)
        F_out = F_SAN + alpha * gated_output
        
        # Apply output normalization
        F_out = self.norm_out[fused_index](F_out)
        
        # ================================================================
        # Concatenate back with query tokens
        # ================================================================
        x = torch.cat([x[:, :-L, ...], F_out], dim=1)
        return x


# ============================================================
# Registered Head for Custom Activation Fusion
# ============================================================
@MODELS.register_module()
class CustomActivationFusionSANCLIPHead(SideAdapterCLIPHead):
    """SAN head with Custom Activation Fusion.
    
    Implements:
        A = Softmax((QK^T / sqrt(d)) * tau + beta * D)
        D = |F_SAN - F_CLIP|
        O = A * V
        G = sigmoid(W_g * F_CLIP)
        F_out = F_SAN + alpha * (G ⊙ O)
    
    Config example:
        decode_head=dict(
            type='CustomActivationFusionSANCLIPHead',
            san_cfg=dict(
                ...
                init_tau=1.0,      # Temperature
                init_beta=0.1,     # Difference weight
                init_alpha=0.5,    # Output scaling
                attn_heads=4,      # Attention heads
            ),
            ...
        )
    """

    def __init__(
        self,
        num_classes,
        san_cfg,
        maskgen_cfg,
        deep_supervision_idxs,
        train_cfg,
        **kwargs
    ):
        # Extract custom activation fusion parameters
        init_tau = san_cfg.pop('init_tau', 1.0)
        init_beta = san_cfg.pop('init_beta', 0.1)
        init_alpha = san_cfg.pop('init_alpha', 0.5)
        attn_heads = san_cfg.pop('attn_heads', 4)
        
        # Initialize base class (skip SideAdapterCLIPHead's __init__)
        super(SideAdapterCLIPHead, self).__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs
        )
        
        assert san_cfg.num_queries == maskgen_cfg.sos_token_num
        del self.conv_seg
        
        # Use CustomActivationFusionSAN
        self.side_adapter_network = CustomActivationFusionSAN(
            init_tau=init_tau,
            init_beta=init_beta,
            init_alpha=init_alpha,
            attn_heads=attn_heads,
            **san_cfg
        )
        
        self.rec_with_attnbias = RecWithAttnbias(**maskgen_cfg)
        self.deep_supervision_idxs = deep_supervision_idxs
        self.train_cfg = train_cfg
        
        if train_cfg:
            from mmseg.utils import MatchMasks
            self.match_masks = MatchMasks(
                num_points=train_cfg.num_points,
                num_queries=san_cfg.num_queries,
                num_classes=num_classes,
                assigner=train_cfg.assigner
            )
