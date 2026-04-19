# Swin-Base + UPerNet for PlantSeg115
# Pretrained on ImageNet-22K, fine-tuned on ADE20K -> fine-tune on PlantSeg115
# Base: 50.13 mIoU on ADE20K (150 classes) -> should transfer well to PlantSeg (116 classes)
#
# Using OpenMMLab pretrained model (backbone + UPerNet head)
# Only classification heads retrained for 116 classes

_base_ = [
    '../_base_/datasets/plantseg115.py',
    '../_base_/default_runtime.py',
]

# =====================================================================
# Data (server paths)
# =====================================================================
crop_size = (512, 512)
data_root = '/home/btech/2023/piyush.kumar23b/data/plantseg'

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='PlantSeg115Dataset',
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='RandomResize',
                scale=(2048, 512),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PlantSeg115Dataset',
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# =====================================================================
# Model: Swin-Base (window7, 224, ImageNet-22K) + UPerNet
# =====================================================================
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# Load from OpenMMLab ADE20K-trained Swin-B UPerNet checkpoint
# This gives us pretrained backbone + UPerNet head (except final classifier)
load_from = '/home/btech/2023/piyush.kumar23b/PlantSeg/pretrained/swin_base_upernet_ade20k_22k.pth'

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=128,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=116,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=116,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# =====================================================================
# Optimizer (following official Swin + UPerNet settings)
# =====================================================================
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    accumulative_counts=4,
    clip_grad=dict(max_norm=1.0, norm_type=2))

# =====================================================================
# LR Schedule
# =====================================================================
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False,
         begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False,
         begin=1500, end=160000)
]

# =====================================================================
# Training
# =====================================================================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        max_keep_ckpts=3,
        save_best=['mIoU', 'mAcc'],
        rule='greater',
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

find_unused_parameters = False
