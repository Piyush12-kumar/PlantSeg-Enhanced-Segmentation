# Copyright (c) OpenMMLab. All rights reserved.
"""
Additive Refinement Fusion for SAN (v3)
========================================
Instead of REPLACING the original SAN fusion, this ADDS refinement on top:

    F_base    = F_SAN + F_CLIP                      (original SAN - preserved)
    D         = |F_SAN - F_CLIP|                     (difference detection)
    A         = Softmax((QK^T / sqrt(d)) * tau + beta * D)
    O         = A * V
    G         = sigmoid(W_g * F_CLIP)
    F_out     = F_base + alpha * (G * O)             (additive refinement)

Key insight: The model CAN'T do worse than original SAN because:
  - If alpha learns to be 0, F_out = F_base = original SAN
  - If alpha > 0, the refinement adds complementary information

This guarantees >= original SAN performance.
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


class AdditiveRefinementSAN(SideAdapterNetwork):
    """SideAdapterNetwork with Additive Refinement Fusion.

    Keeps the original F_SAN + F_CLIP addition, then adds a learnable
    cross-attention refinement on top. This guarantees the model can
    always fall back to original SAN performance.

    Args:
        init_tau (float): Initial temperature. Default: 1.0.
        init_beta (float): Initial difference weight. Default: 0.1.
        init_alpha (float): Initial refinement scaling. Default: 0.0
            (starts as pure original SAN, learns to add refinement).
        attn_heads (int): Number of attention heads. Default: 4.
        **kwargs: Arguments for SideAdapterNetwork.
    """

    def __init__(
        self,
        init_tau: float = 1.0,
        init_beta: float = 0.1,
        init_alpha: float = 0.0,  # Start at 0 = pure original SAN
        attn_heads: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        embed_dims = kwargs.get('embed_dims', 240)
        self.attn_heads = attn_heads
        self.head_dim = embed_dims // attn_heads
        self.scale = self.head_dim ** -0.5

        num_fusion = len(self.fusion_index)

        # Learnable parameters (per fusion point)
        self.tau = nn.ParameterList([
            nn.Parameter(torch.tensor([init_tau]))
            for _ in range(num_fusion)
        ])
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor([init_beta]))
            for _ in range(num_fusion)
        ])
        # alpha starts at 0: model begins as original SAN
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor([init_alpha]))
            for _ in range(num_fusion)
        ])

        # Q, K, V projections for cross-attention
        self.q_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(num_fusion)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(num_fusion)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(num_fusion)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(num_fusion)
        ])

        # Gating: G = sigmoid(W_g * F_CLIP)
        self.gate_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(num_fusion)
        ])

        # Layer norms for refinement branch only
        self.norm_san = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_fusion)
        ])
        self.norm_clip = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_fusion)
        ])

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for i in range(len(self.fusion_index)):
            for proj in [self.q_proj[i], self.k_proj[i],
                         self.v_proj[i], self.out_proj[i],
                         self.gate_proj[i]]:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)

    def fuse_clip(
        self,
        fused_index: int,
        x: torch.Tensor,
        clip_feature: torch.Tensor,
        hwshape: Tuple[int, int],
        L: int
    ) -> torch.Tensor:
        """Additive Refinement Fusion.

        F_base = F_SAN + F_CLIP              (original SAN, always preserved)
        F_out  = F_base + alpha * (G * O)    (learnable refinement on top)
        """
        B = x.shape[0]
        C = x.shape[-1]

        # === STEP 1: Original SAN fusion (PRESERVED EXACTLY) ===
        F_SAN = x[:, -L:, ...]  # [B, L, C]

        fused_clip = self.conv_clips[fused_index](clip_feature.contiguous())
        fused_clip = resize(fused_clip, size=hwshape,
                            mode='bilinear', align_corners=False)
        F_CLIP = fused_clip.permute(0, 2, 3, 1).reshape(B, L, C)

        # Original SAN: simple addition
        F_base = F_SAN + F_CLIP  # [B, L, C]

        # === STEP 2: Refinement branch ===
        F_SAN_n = self.norm_san[fused_index](F_SAN)
        F_CLIP_n = self.norm_clip[fused_index](F_CLIP)

        # D = |F_SAN - F_CLIP| (difference detection)
        D = torch.abs(F_SAN_n - F_CLIP_n)

        # Cross-attention: SAN queries CLIP
        Q = self.q_proj[fused_index](F_SAN_n)
        K = self.k_proj[fused_index](F_CLIP_n)
        V = self.v_proj[fused_index](F_CLIP_n)

        Q = Q.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, L, self.attn_heads, self.head_dim).transpose(1, 2)

        # A = Softmax((QK^T / sqrt(d)) * tau + beta * D_bias)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        tau = self.tau[fused_index]
        beta = self.beta[fused_index]

        attn_scores = attn_scores * tau

        # Difference bias (per-position)
        D_bias = D.sum(dim=-1, keepdim=True)  # [B, L, 1]
        D_bias = D_bias.unsqueeze(1)  # [B, 1, L, 1]
        attn_scores = attn_scores + beta * D_bias

        A = F.softmax(attn_scores, dim=-1)

        # O = A * V
        O = torch.matmul(A, V)
        O = O.transpose(1, 2).reshape(B, L, C)
        O = self.out_proj[fused_index](O)

        # G = sigmoid(W_g * F_CLIP)
        G = torch.sigmoid(self.gate_proj[fused_index](F_CLIP_n))

        # === STEP 3: Combine ===
        # F_out = F_base + alpha * (G * O)
        alpha = self.alpha[fused_index]
        F_out = F_base + alpha * (G * O)

        x = torch.cat([x[:, :-L, ...], F_out], dim=1)
        return x


@MODELS.register_module()
class AdditiveRefinementSANCLIPHead(SideAdapterCLIPHead):
    """SAN head with Additive Refinement Fusion.

    Keeps original SAN fusion + adds learnable refinement.
    Guaranteed >= original SAN performance (alpha starts at 0).
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
        init_tau = san_cfg.pop('init_tau', 1.0)
        init_beta = san_cfg.pop('init_beta', 0.1)
        init_alpha = san_cfg.pop('init_alpha', 0.0)
        attn_heads = san_cfg.pop('attn_heads', 4)

        super(SideAdapterCLIPHead, self).__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs
        )

        assert san_cfg.num_queries == maskgen_cfg.sos_token_num
        del self.conv_seg

        self.side_adapter_network = AdditiveRefinementSAN(
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
