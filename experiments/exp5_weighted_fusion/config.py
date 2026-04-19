# ============================================================
# Experiment 5: Weighted (Gated) Feature Fusion
# Hypothesis: Learnable per-layer gating weights with channel
# attention allow adaptive control of CLIP feature contribution,
# preventing information overload in early layers.
# Changes: decode_head type → WeightedFusionSANCLIPHead
# ============================================================

_base_ = [
    '../../configs/_base_/models/san_vit-b16.py',
    '../../configs/_base_/default_runtime.py',
]

# --------------- Dataset Settings ---------------
dataset_type = 'PlantSeg115Dataset'
data_root = '/home/btech/2023/piyush.kumar23b/data/plantseg'
crop_size = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# --------------- Model Settings (ViT-L/14 + Weighted Fusion) ---------------
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth'

data_preprocessor = dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=640,
    test_cfg=dict(size_divisor=32))

model = dict(
    pretrained=pretrained,
    encoder_resolution=0.7,
    image_encoder=dict(
        type='VisionTransformer',
        img_size=(336, 336),
        patch_size=14,
        patch_pad=0,
        embed_dims=1024,
        num_layers=18,
        num_heads=16,
        out_indices=(5, 11, 17),
    ),
    text_encoder=dict(
        dataset_name='plantseg115',
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        output_dims=768,
    ),
    decode_head=dict(
        type='WeightedFusionSANCLIPHead',
        num_classes=116,
        san_cfg=dict(
            clip_channels=1024,
            cfg_decoder=dict(num_heads=16)),
        maskgen_cfg=dict(
            num_layers=6,
            embed_dims=1024,
            num_heads=16,
            out_dims=768,
        )))

# --------------- Training Schedule ---------------
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=10000,
    val_begin=10000)

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

# --------------- Optimizer (same as baseline) ---------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

# --------------- LR Scheduler ---------------
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False,
    )
]
