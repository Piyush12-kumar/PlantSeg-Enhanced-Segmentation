"""
BiFormer: Vision Transformer with Bi-Level Routing Attention (CVPR 2023)

Self-contained implementation for mmseg 1.2.2.
Adapted from: https://github.com/rayleizhu/BiFormer

Key idea: Bi-Level Routing Attention (BRA) routes tokens to relevant regions
first (coarse level), then applies fine-grained attention only within those
regions. This is more efficient and effective than standard global attention.
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log

from mmseg.registry import MODELS


# =====================================================================
# Core BRA Components (from ops/bra_legacy.py)
# =====================================================================

class TopkRouting(nn.Module):
    """Differentiable top-k routing with scaling.
    
    Routes each region to its top-k most relevant regions based on
    region-level query-key similarity.
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None,
                 param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key: (n, p^2, c) region-level features
        Returns:
            r_weight: (n, p^2, topk) routing weights
            topk_index: (n, p^2, topk) routing indices
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index


class KVGather(nn.Module):
    """Gather key-value pairs based on routing indices."""
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: torch.Tensor, r_weight: torch.Tensor,
                kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r_idx: (n, p^2, topk) routing indices
            r_weight: (n, p^2, topk) routing weights
            kv: (n, p^2, w^2, c_kq+c_v)
        Returns:
            (n, p^2, topk, w^2, c_kq+c_v) gathered kv
        """
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
        )
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        return topk_kv


class QKVLinear(nn.Module):
    """Linear projection for Q, KV (joint K and V)."""
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """Bi-Level Routing Attention (BRA).
    
    Level 1 (Region): Routes each region to top-k most relevant regions.
    Level 2 (Token): Applies token-to-token attention within selected regions.
    
    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        n_win: Number of windows per side.
        topk: Number of top-k regions to attend to.
        side_dwconv: Kernel size for locality-enhanced position encoding.
        auto_pad: Whether to auto-pad for non-divisible sizes.
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None,
                 kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False,
                 diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0
        self.scale = qk_scale or self.qk_dim ** -0.5

        # Locality-Enhanced Position Encoding (LePE)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1,
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        # Routing
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing

        assert not (self.param_routing and not self.diff_routing)
        self.router = TopkRouting(
            qk_dim=self.qk_dim, qk_scale=self.scale, topk=self.topk,
            diff_routing=self.diff_routing, param_routing=self.param_routing)

        if self.soft_routing:
            mul_weight = 'soft'
        elif self.diff_routing:
            mul_weight = 'hard'
        else:
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # QKV projections
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} not supported')

        # KV downsampling
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        if self.kv_downsample_mode == 'ada_avgpool':
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        else:
            raise ValueError(f'kv_downsample_mode {self.kv_downsample_mode} not supported')

        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        Args:
            x: NHWC tensor
        Returns:
            NHWC tensor
        """
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.size()
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0

        # Partition into windows: (n, p^2, w, w, c)
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c",
                       j=self.n_win, i=self.n_win)

        # QKV projection
        q, kv = self.qkv(x)

        # Pixel-level q, kv
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c',
                           j=self.n_win, i=self.n_win)

        # Region-level routing
        q_win = q.mean([2, 3])
        k_win = kv[..., 0:self.qk_dim].mean([2, 3])

        # LePE (Locality-Enhanced Position Encoding)
        lepe = self.lepe(
            rearrange(kv[..., self.qk_dim:],
                      'n (j i) h w c -> n c (j h) (i w)',
                      j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c',
                         j=self.n_win, i=self.n_win)

        # Gather top-k kv
        r_weight, r_idx = self.router(q_win, k_win)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        # Multi-head attention
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)',
                        j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        out = self.wo(out)

        # Remove padding
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out


# =====================================================================
# Common Modules
# =====================================================================

class DWConv(nn.Module):
    """Depthwise convolution operating on NHWC tensor."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Attention(nn.Module):
    """Vanilla multi-head attention (for topk=-1)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x


class AttentionLePE(nn.Module):
    """Vanilla attention with LePE (for topk=-2)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1,
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

    def forward(self, x):
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x


# =====================================================================
# BiFormer Block & Backbone
# =====================================================================

class BiFormerBlock(nn.Module):
    """Single BiFormer block with BRA attention + FFN."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4,
                 kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False,
                 diff_routing=False, soft_routing=False, mlp_ratio=4,
                 mlp_dwconv=False, side_dwconv=5, before_attn_dwconv=3,
                 pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # Position embedding via DWConv
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv,
                                       padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        # Attention: BRA (topk>0), vanilla (topk=-1), vanilla+LePE (topk=-2), conv (topk=0)
        if topk > 0:
            self.attn = BiLevelRoutingAttention(
                dim=dim, num_heads=num_heads, n_win=n_win,
                qk_dim=qk_dim, qk_scale=qk_scale,
                kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                kv_downsample_kernel=kv_downsample_kernel,
                kv_downsample_mode=kv_downsample_mode,
                topk=topk, param_attention=param_attention,
                param_routing=param_routing, diff_routing=diff_routing,
                soft_routing=soft_routing, side_dwconv=side_dwconv,
                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            from einops.layers.torch import Rearrange
            self.attn = nn.Sequential(
                Rearrange('n h w c -> n c h w'),
                nn.Conv2d(dim, dim, 1),
                nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
                nn.Conv2d(dim, dim, 1),
                Rearrange('n c h w -> n h w c'))

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Layer scale
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """x: NCHW tensor"""
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC

        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))
                x = self.norm2(x + self.drop_path(self.mlp(x)))

        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm for NCHW tensors (from timm)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# =====================================================================
# BiFormer Backbone (registered for mmseg)
# =====================================================================

@MODELS.register_module()
class BiFormerBackbone(BaseModule):
    """BiFormer backbone for semantic segmentation.
    
    A hierarchical vision transformer with Bi-Level Routing Attention.
    Outputs multi-scale features from 4 stages (stride 4, 8, 16, 32).
    
    Args:
        depth (list[int]): Number of blocks in each stage.
        embed_dim (list[int]): Embedding dimension for each stage.
        mlp_ratios (list[int]): MLP expansion ratio per stage.
        n_win (int): Number of windows per side for BRA.
        topks (list[int]): Top-k values for routing per stage.
        head_dim (int): Dimension per attention head.
        drop_path_rate (float): Stochastic depth rate.
        pretrained (str): Path to pretrained checkpoint.
        auto_pad (bool): Whether to auto-pad for any input size.
    """
    def __init__(self,
                 depth=[3, 4, 8, 3],
                 in_chans=3,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 qk_scale=None,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 use_checkpoint_stages=[],
                 # BRA parameters
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False,
                 diff_routing=False,
                 soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False,
                 # mmseg params
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.num_features = self.embed_dim = embed_dim
        self.pretrained = pretrained

        # ---- Downsample layers (patch embeddings) ----
        self.downsample_layers = nn.ModuleList()
        # Stem: 2x stride-2 convolutions = overall stride 4
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            self.downsample_layers.append(downsample_layer)

        # ---- Transformer stages ----
        qk_dims = [d if d is not None else embed_dim[i] for i, d in enumerate(qk_dims)]
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                BiFormerBlock(
                    dim=embed_dim[i], drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    topk=topks[i], num_heads=nheads[i], n_win=n_win,
                    qk_dim=qk_dims[i], qk_scale=qk_scale,
                    kv_per_win=kv_per_wins[i],
                    kv_downsample_ratio=kv_downsample_ratios[i],
                    kv_downsample_kernel=kv_downsample_kernels[i],
                    kv_downsample_mode=kv_downsample_mode,
                    param_attention=param_attention,
                    param_routing=param_routing,
                    diff_routing=diff_routing, soft_routing=soft_routing,
                    mlp_ratio=mlp_ratios[i], mlp_dwconv=mlp_dwconv,
                    side_dwconv=side_dwconv,
                    before_attn_dwconv=before_attn_dwconv,
                    pre_norm=pre_norm, auto_pad=auto_pad)
                for j in range(depth[i])
            ])
            self.stages.append(stage)
            cur += depth[i]

        # ---- Extra norms for dense prediction ----
        self.extra_norms = nn.ModuleList([
            LayerNorm2d(embed_dim[i]) for i in range(4)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        """Load pretrained weights."""
        if self.pretrained is not None:
            print_log(f'Loading BiFormer pretrained from {self.pretrained}',
                      logger='current')
            ckpt = load_checkpoint(self, self.pretrained,
                                   map_location='cpu', strict=False)
        elif self.init_cfg is not None:
            super().init_weights()

    def forward(self, x):
        """Extract multi-scale features.
        
        Returns:
            tuple[Tensor]: 4 feature maps at stride 4, 8, 16, 32.
        """
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(self.extra_norms[i](x))
        return tuple(outs)
