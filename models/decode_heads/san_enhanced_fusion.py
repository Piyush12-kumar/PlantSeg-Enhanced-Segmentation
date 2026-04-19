# Copyright (c) OpenMMLab. All rights reserved.
"""Enhanced SAN Feature Fusion with:
1. Learnable temperature for attention scaling
2. Gated attention mechanism
3. Feature difference term for complementary information

This module provides an advanced fusion strategy that adaptively combines
CLIP features with spatial visual tokens.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model.weight_init import caffe2_xavier_init, trunc_normal_

from mmseg.registry import MODELS
from ..utils import LayerNorm2d, resize
from .san_head import (MLPMaskDecoder, RecWithAttnbias,
                       SideAdapterCLIPHead, SideAdapterNetwork)


class EnhancedFusionSAN(SideAdapterNetwork):
    """SideAdapterNetwork with Enhanced Feature Fusion.
    
    Incorporates three key improvements over the basic additive fusion:
    
    1. Learnable Temperature: Scales attention logits with a learnable
       temperature parameter for sharper/softer attention distributions.
       
    2. Gated Attention: Uses a learnable gate to control the flow of
       CLIP information into spatial features, allowing the model to
       learn when and how much CLIP features should contribute.
       
    3. Feature Difference Term: Captures complementary information by
       computing the difference between spatial and CLIP features,
       allowing the model to learn from both similarities and differences.
    
    Args:
        init_temperature (float): Initial value for learnable temperature.
            Lower values produce sharper attention. Default: 1.0.
        attn_heads (int): Number of attention heads. Default: 4.
        use_feature_diff (bool): Whether to use feature difference term.
            Default: True.
        diff_weight_init (float): Initial weight for the difference term.
            Default: 0.1.
        gate_init (float): Initial value for fusion gates (before sigmoid).
            Default: 0.0 (sigmoid(0.0) = 0.5).
        **kwargs: Additional arguments passed to SideAdapterNetwork.
    """

    def __init__(
        self,
        init_temperature: float = 1.0,
        attn_heads: int = 4,
        use_feature_diff: bool = True,
        diff_weight_init: float = 0.1,
        gate_init: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        embed_dims = kwargs.get('embed_dims', 240)
        self.attn_heads = attn_heads
        self.head_dim = embed_dims // attn_heads
        self.use_feature_diff = use_feature_diff
        
        # ================================================================
        # 1. Learnable Temperature Parameters (one per fusion point)
        # ================================================================
        # Temperature controls attention sharpness: lower = sharper focus
        # Initialize with log(temp) for numerical stability during training
        self.temperature_log = nn.ParameterList([
            nn.Parameter(torch.tensor([float(init_temperature)]).log())
            for _ in range(len(self.fusion_index))
        ])
        
        # ================================================================
        # 2. Gated Attention Mechanism
        # ================================================================
        # Cross-attention layers with gating
        self.cross_attn_layers = nn.ModuleList()
        self.attn_norms_q = nn.ModuleList()  # Query normalization
        self.attn_norms_k = nn.ModuleList()  # Key normalization
        
        for _ in range(len(self.fusion_index)):
            # Query, Key, Value projections for multi-head attention
            self.cross_attn_layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(embed_dims, embed_dims),
                'k_proj': nn.Linear(embed_dims, embed_dims),
                'v_proj': nn.Linear(embed_dims, embed_dims),
                'out_proj': nn.Linear(embed_dims, embed_dims),
            }))
            self.attn_norms_q.append(nn.LayerNorm(embed_dims))
            self.attn_norms_k.append(nn.LayerNorm(embed_dims))
        
        # Learnable fusion gates (one per fusion point)
        # Initialized to gate_init, sigmoid produces ~0.5 when gate_init=0
        self.fusion_gates = nn.ParameterList([
            nn.Parameter(torch.tensor([gate_init]))
            for _ in range(len(self.fusion_index))
        ])
        
        # Content-adaptive gating network
        # Takes both spatial and CLIP features to compute dynamic gate
        self.gate_networks = nn.ModuleList()
        for _ in range(len(self.fusion_index)):
            self.gate_networks.append(nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.GELU(),
                nn.Linear(embed_dims, embed_dims),
                nn.Sigmoid()
            ))
        
        # ================================================================
        # 3. Feature Difference Term
        # ================================================================
        if self.use_feature_diff:
            # Learnable weights for the difference term (per fusion point)
            self.diff_weights = nn.ParameterList([
                nn.Parameter(torch.tensor([diff_weight_init]))
                for _ in range(len(self.fusion_index))
            ])
            # Transform for difference features
            self.diff_transforms = nn.ModuleList()
            for _ in range(len(self.fusion_index)):
                self.diff_transforms.append(nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.GELU(),
                    nn.Linear(embed_dims, embed_dims)
                ))
        
        # Final fusion layer normalization
        self.fusion_norms = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(len(self.fusion_index))
        ])

    def init_weights(self):
        """Initialize weights for enhanced fusion components."""
        super().init_weights()
        
        for i in range(len(self.fusion_index)):
            # Initialize attention projections
            for name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                layer = self.cross_attn_layers[i][name]
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            
            # Initialize layer norms
            nn.init.ones_(self.attn_norms_q[i].weight)
            nn.init.zeros_(self.attn_norms_q[i].bias)
            nn.init.ones_(self.attn_norms_k[i].weight)
            nn.init.zeros_(self.attn_norms_k[i].bias)
            nn.init.ones_(self.fusion_norms[i].weight)
            nn.init.zeros_(self.fusion_norms[i].bias)
            
            # Initialize gate networks
            for m in self.gate_networks[i].modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            
            # Initialize difference transforms
            if self.use_feature_diff:
                for m in self.diff_transforms[i].modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """Compute scaled dot-product attention with learnable temperature.
        
        Args:
            query: [B, num_heads, L_q, head_dim]
            key: [B, num_heads, L_k, head_dim]
            value: [B, num_heads, L_k, head_dim]
            temperature: Scalar learnable temperature
            
        Returns:
            Attention output: [B, num_heads, L_q, head_dim]
        """
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply learnable temperature scaling
        # temperature > 1 makes distribution softer, < 1 makes it sharper
        attn_scores = attn_scores / temperature.exp().clamp(min=1e-6)
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        return output

    def fuse_clip(
        self,
        fused_index: int,
        x: torch.Tensor,
        clip_feature: torch.Tensor,
        hwshape: Tuple[int, int],
        L: int
    ) -> torch.Tensor:
        """Enhanced CLIP-spatial feature fusion.
        
        Implements:
        1. Cross-attention with learnable temperature
        2. Gated fusion (both static and content-adaptive)
        3. Feature difference term
        
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
        
        # Project CLIP features to embedding dimension
        fused_clip = self.conv_clips[fused_index](clip_feature.contiguous())
        fused_clip = resize(fused_clip, size=hwshape,
                           mode='bilinear', align_corners=False)
        # [B, C, H, W] -> [B, H*W, C]
        clip_tokens = fused_clip.permute(0, 2, 3, 1).reshape(B, L, C)
        
        # Extract spatial tokens from x
        spatial = x[:, -L:, ...]  # [B, L, C]
        
        # ================================================================
        # Step 1: Cross-Attention with Learnable Temperature
        # ================================================================
        # Normalize inputs
        spatial_norm = self.attn_norms_q[fused_index](spatial)
        clip_norm = self.attn_norms_k[fused_index](clip_tokens)
        
        # Get projections
        attn_module = self.cross_attn_layers[fused_index]
        Q = attn_module['q_proj'](spatial_norm)  # [B, L, C]
        K = attn_module['k_proj'](clip_norm)      # [B, L, C]
        V = attn_module['v_proj'](clip_norm)      # [B, L, C]
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention with learnable temperature
        temperature = self.temperature_log[fused_index]
        attn_out = self._scaled_dot_product_attention(Q, K, V, temperature)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)
        attn_out = attn_module['out_proj'](attn_out)
        
        # ================================================================
        # Step 2: Gated Fusion
        # ================================================================
        # Static learnable gate
        static_gate = torch.sigmoid(self.fusion_gates[fused_index])
        
        # Content-adaptive gate based on both features
        # Average pool for global context
        spatial_avg = spatial.mean(dim=1)  # [B, C]
        clip_avg = clip_tokens.mean(dim=1)  # [B, C]
        combined = torch.cat([spatial_avg, clip_avg], dim=-1)  # [B, 2C]
        
        # Compute channel-wise adaptive gate
        adaptive_gate = self.gate_networks[fused_index](combined)  # [B, C]
        adaptive_gate = adaptive_gate.unsqueeze(1)  # [B, 1, C]
        
        # Combine static and adaptive gates
        # The static gate controls overall flow, adaptive gate refines per-channel
        combined_gate = static_gate * adaptive_gate
        
        # Apply gated attention output
        gated_attn = combined_gate * attn_out
        
        # ================================================================
        # Step 3: Feature Difference Term
        # ================================================================
        if self.use_feature_diff:
            # Compute difference between spatial and CLIP features
            # This captures complementary information
            feature_diff = spatial - clip_tokens  # [B, L, C]
            
            # Transform the difference
            diff_transformed = self.diff_transforms[fused_index](feature_diff)
            
            # Weight the difference contribution
            diff_weight = self.diff_weights[fused_index]
            diff_term = diff_weight * diff_transformed
        else:
            diff_term = 0
        
        # ================================================================
        # Final Fusion: spatial + gated_attention + difference_term
        # ================================================================
        fused = spatial + gated_attn + diff_term
        fused = self.fusion_norms[fused_index](fused)
        
        # Concatenate with query tokens
        x = torch.cat([x[:, :-L, ...], fused], dim=1)
        return x


# ============================================================
# Registered Head for Enhanced Fusion
# ============================================================
@MODELS.register_module()
class EnhancedFusionSANCLIPHead(SideAdapterCLIPHead):
    """SAN head with enhanced feature fusion.
    
    Features:
    - Learnable temperature for attention scaling
    - Gated attention with static and content-adaptive gates
    - Feature difference term for complementary information
    
    Config example:
        decode_head=dict(
            type='EnhancedFusionSANCLIPHead',
            san_cfg=dict(
                ...
                init_temperature=1.0,
                attn_heads=4,
                use_feature_diff=True,
                diff_weight_init=0.1,
                gate_init=0.0,
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
        # Extract enhanced fusion specific parameters
        init_temperature = san_cfg.pop('init_temperature', 1.0)
        attn_heads = san_cfg.pop('attn_heads', 4)
        use_feature_diff = san_cfg.pop('use_feature_diff', True)
        diff_weight_init = san_cfg.pop('diff_weight_init', 0.1)
        gate_init = san_cfg.pop('gate_init', 0.0)
        
        # Initialize base class (skip SideAdapterCLIPHead's __init__)
        super(SideAdapterCLIPHead, self).__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs
        )
        
        assert san_cfg.num_queries == maskgen_cfg.sos_token_num
        del self.conv_seg
        
        # Use EnhancedFusionSAN instead of default SideAdapterNetwork
        self.side_adapter_network = EnhancedFusionSAN(
            init_temperature=init_temperature,
            attn_heads=attn_heads,
            use_feature_diff=use_feature_diff,
            diff_weight_init=diff_weight_init,
            gate_init=gate_init,
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
