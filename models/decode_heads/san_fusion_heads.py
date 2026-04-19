# Copyright (c) OpenMMLab. All rights reserved.
"""Modified SAN heads with improved CLIP-visual feature fusion mechanisms.

Three variants replacing the simple addition in SideAdapterNetwork.fuse_clip():
1. AttentionFusionSAN: Cross-attention between CLIP and spatial features
2. WeightedFusionSAN: Learnable per-layer gated fusion weights
3. MultiScaleFusionSAN: Multi-scale CLIP feature aggregation before fusion
"""
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model.weight_init import caffe2_xavier_init, trunc_normal_

from mmseg.registry import MODELS
from ..utils import LayerNorm2d, resize
from .san_head import (MLPMaskDecoder, RecWithAttnbias,
                       SideAdapterCLIPHead, SideAdapterNetwork)


# ============================================================
# Experiment 4: Attention-Based Fusion
# ============================================================
class AttentionFusionSAN(SideAdapterNetwork):
    """SideAdapterNetwork with cross-attention based CLIP fusion.

    Instead of simple addition, uses a lightweight cross-attention
    where spatial tokens attend to projected CLIP features.
    """

    def __init__(self, attn_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        embed_dims = kwargs.get('embed_dims', 240)
        self.attn_heads = attn_heads
        # Cross-attention layers (one per fusion point)
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(len(self.fusion_index)):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dims,
                    num_heads=attn_heads,
                    batch_first=True,
                    dropout=0.0))
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(len(self.fusion_index))
        ])

    def init_weights(self):
        super().init_weights()
        for layer in self.cross_attn_layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for norm in self.attn_norms:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def fuse_clip(self, fused_index, x, clip_feature, hwshape, L):
        """Fuse via cross-attention: spatial tokens attend to CLIP features."""
        fused_clip = self.conv_clips[fused_index](
            clip_feature.contiguous())
        fused_clip = resize(
            fused_clip, size=hwshape,
            mode='bilinear', align_corners=False)
        # [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = fused_clip.shape
        clip_tokens = fused_clip.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # Spatial tokens from x
        spatial = x[:, -L:, ...]
        # Cross-attention: spatial queries attend to CLIP key/values
        spatial_normed = self.attn_norms[fused_index](spatial)
        attn_out, _ = self.cross_attn_layers[fused_index](
            query=spatial_normed, key=clip_tokens, value=clip_tokens)
        # Residual connection
        spatial = spatial + attn_out
        x = torch.cat([x[:, :-L, ...], spatial], dim=1)
        return x


# ============================================================
# Experiment 5: Weighted Feature Fusion
# ============================================================
class WeightedFusionSAN(SideAdapterNetwork):
    """SideAdapterNetwork with learnable gated fusion weights.

    Each fusion point has a learnable scalar gate (sigmoid-activated)
    controlling how much CLIP information flows into spatial tokens.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embed_dims = kwargs.get('embed_dims', 240)
        # Learnable fusion gate per fusion point
        self.fusion_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(len(self.fusion_index))
        ])
        # Channel attention for adaptive weighting
        self.channel_attn = nn.ModuleList()
        for _ in range(len(self.fusion_index)):
            self.channel_attn.append(nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(embed_dims, embed_dims // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims // 4, embed_dims),
                nn.Sigmoid()))

    def init_weights(self):
        super().init_weights()
        for gate in self.fusion_gates:
            nn.init.zeros_(gate)
        for ca in self.channel_attn:
            for m in ca.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def fuse_clip(self, fused_index, x, clip_feature, hwshape, L):
        """Fuse with learnable gate and channel attention."""
        fused_clip = self.conv_clips[fused_index](
            clip_feature.contiguous())
        fused_clip = resize(
            fused_clip, size=hwshape,
            mode='bilinear', align_corners=False)
        fused_clip = fused_clip.permute(0, 2, 3, 1).reshape(
            x[:, -L:, ...].shape)
        # Compute channel attention weights
        spatial = x[:, -L:, ...]  # [B, L, C]
        ca_weight = self.channel_attn[fused_index](
            spatial.permute(0, 2, 1))  # [B, C]
        ca_weight = ca_weight.unsqueeze(1)  # [B, 1, C]
        # Apply gated fusion
        gate = torch.sigmoid(self.fusion_gates[fused_index])
        fused = spatial + gate * (ca_weight * fused_clip)
        x = torch.cat([x[:, :-L, ...], fused], dim=1)
        return x


# ============================================================
# Experiment 6: Multi-Scale Fusion
# ============================================================
class MultiScaleFusionSAN(SideAdapterNetwork):
    """SideAdapterNetwork with multi-scale CLIP feature aggregation.

    Processes CLIP features at multiple spatial scales (1x, 0.5x)
    and combines them with learned weights before fusing.
    """

    def __init__(self, scales: list = [1.0, 0.5], **kwargs):
        super().__init__(**kwargs)
        embed_dims = kwargs.get('embed_dims', 240)
        self.scales = scales
        num_scales = len(scales)
        # Scale mixing weights (per fusion point)
        self.scale_weights = nn.ParameterList([
            nn.Parameter(torch.ones(num_scales) / num_scales)
            for _ in range(len(self.fusion_index))
        ])
        # Extra projection for each scale beyond the first
        self.extra_projs = nn.ModuleList()
        for _ in range(len(self.fusion_index)):
            projs = nn.ModuleList()
            for s_idx in range(1, num_scales):
                projs.append(nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, 3, padding=1,
                              groups=embed_dims),
                    nn.Conv2d(embed_dims, embed_dims, 1),
                    nn.GELU()))
            self.extra_projs.append(projs)

    def init_weights(self):
        super().init_weights()
        for projs in self.extra_projs:
            for proj in projs:
                for m in proj.modules():
                    if isinstance(m, nn.Conv2d):
                        caffe2_xavier_init(m)

    def fuse_clip(self, fused_index, x, clip_feature, hwshape, L):
        """Fuse CLIP features processed at multiple scales."""
        clip_channels = self.conv_clips[fused_index](
            clip_feature.contiguous())
        # Process at multiple scales
        scale_features = []
        weights = torch.softmax(self.scale_weights[fused_index], dim=0)
        for s_idx, scale in enumerate(self.scales):
            if scale == 1.0:
                feat = resize(clip_channels, size=hwshape,
                              mode='bilinear', align_corners=False)
            else:
                scaled_size = (max(1, int(hwshape[0] * scale)),
                               max(1, int(hwshape[1] * scale)))
                feat = resize(clip_channels, size=scaled_size,
                              mode='bilinear', align_corners=False)
                if s_idx > 0:
                    feat = self.extra_projs[fused_index][s_idx - 1](feat)
                feat = resize(feat, size=hwshape,
                              mode='bilinear', align_corners=False)
            scale_features.append(feat * weights[s_idx])
        # Aggregate multi-scale features
        fused_clip = sum(scale_features)
        fused_clip = fused_clip.permute(0, 2, 3, 1).reshape(
            x[:, -L:, ...].shape)
        x = torch.cat([x[:, :-L, ...], x[:, -L:, ...] + fused_clip], dim=1)
        return x


# ============================================================
# Registered Head Variants
# ============================================================
@MODELS.register_module()
class AttentionFusionSANCLIPHead(SideAdapterCLIPHead):
    """SAN head with attention-based CLIP fusion."""

    def __init__(self, num_classes, san_cfg, maskgen_cfg,
                 deep_supervision_idxs, train_cfg, **kwargs):
        # Extract attn_heads before passing to parent
        attn_heads = san_cfg.pop('attn_heads', 4)
        super(SideAdapterCLIPHead, self).__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs)
        assert san_cfg.num_queries == maskgen_cfg.sos_token_num
        del self.conv_seg
        self.side_adapter_network = AttentionFusionSAN(
            attn_heads=attn_heads, **san_cfg)
        self.rec_with_attnbias = RecWithAttnbias(**maskgen_cfg)
        self.deep_supervision_idxs = deep_supervision_idxs
        self.train_cfg = train_cfg
        if train_cfg:
            from mmseg.utils import MatchMasks
            self.match_masks = MatchMasks(
                num_points=train_cfg.num_points,
                num_queries=san_cfg.num_queries,
                num_classes=num_classes,
                assigner=train_cfg.assigner)


@MODELS.register_module()
class WeightedFusionSANCLIPHead(SideAdapterCLIPHead):
    """SAN head with weighted gated CLIP fusion."""

    def __init__(self, num_classes, san_cfg, maskgen_cfg,
                 deep_supervision_idxs, train_cfg, **kwargs):
        super(SideAdapterCLIPHead, self).__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs)
        assert san_cfg.num_queries == maskgen_cfg.sos_token_num
        del self.conv_seg
        self.side_adapter_network = WeightedFusionSAN(**san_cfg)
        self.rec_with_attnbias = RecWithAttnbias(**maskgen_cfg)
        self.deep_supervision_idxs = deep_supervision_idxs
        self.train_cfg = train_cfg
        if train_cfg:
            from mmseg.utils import MatchMasks
            self.match_masks = MatchMasks(
                num_points=train_cfg.num_points,
                num_queries=san_cfg.num_queries,
                num_classes=num_classes,
                assigner=train_cfg.assigner)


@MODELS.register_module()
class MultiScaleFusionSANCLIPHead(SideAdapterCLIPHead):
    """SAN head with multi-scale CLIP fusion."""

    def __init__(self, num_classes, san_cfg, maskgen_cfg,
                 deep_supervision_idxs, train_cfg, **kwargs):
        scales = san_cfg.pop('scales', [1.0, 0.5])
        super(SideAdapterCLIPHead, self).__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs)
        assert san_cfg.num_queries == maskgen_cfg.sos_token_num
        del self.conv_seg
        self.side_adapter_network = MultiScaleFusionSAN(
            scales=scales, **san_cfg)
        self.rec_with_attnbias = RecWithAttnbias(**maskgen_cfg)
        self.deep_supervision_idxs = deep_supervision_idxs
        self.train_cfg = train_cfg
        if train_cfg:
            from mmseg.utils import MatchMasks
            self.match_masks = MatchMasks(
                num_points=train_cfg.num_points,
                num_queries=san_cfg.num_queries,
                num_classes=num_classes,
                assigner=train_cfg.assigner)
