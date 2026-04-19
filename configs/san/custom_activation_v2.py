# ============================================================
# Custom Activation Fusion SAN ViT-L/14 - TUNED v2
# ============================================================
# Changes from v1 (previous training that got mIoU 27.7):
#
# 1. LR WARMUP: Added LinearLR warmup for first 1500 iters
#    (paper uses this, we didn't before -- causes unstable early training)
#
# 2. LEARNING RATE: Reduced from 1e-4 to 6e-5 (matches SAN paper)
#    Previous was too high, causing oscillation in later stages
#
# 3. WEIGHT DECAY: Increased from 1e-4 to 0.01 (matches SAN paper)
#    Stronger regularization prevents overfitting on 116-class task
#
# 4. GRADIENT CLIPPING: Relaxed from 0.01 to 0.1 (too aggressive before)
#    Previous clipping was choking gradient flow for fusion parameters
#
# 5. FUSION PARAMS LR: Increased lr_mult to 5.0 for tau/beta/alpha
#    These small params need higher LR to learn meaningful values
#
# 6. STRONGER AUGMENTATION: Added RandomRotate, aggressive PhotoMetric
#    More augmentation = better generalization on plant images
#
# 7. WIDER SCALE RANGE: [256..1280] instead of [320..960]
#    Sees more diverse scales during training
#
# 8. INIT VALUES: tau=0.5, beta=0.05, alpha=0.3
#    Previous init was too aggressive (tau=1.0, alpha=0.5)
#    Gentler init lets model warm up before fusion kicks in
#
# 9. LONGER TRAINING: 200k iters (from 160k)
#    v1 was still improving at 160k -- more time to converge
#
# 10. COSINE ANNEALING: Better LR schedule for longer training
#     PolyLR decays too fast; cosine gives better late-stage learning
# ============================================================

_base_ = [
    '../_base_/models/san_vit-b16.py',
    '../_base_/default_runtime.py',
]

# --------------- Dataset Settings ---------------
dataset_type = 'PlantSeg115Dataset'
data_root = '/home/btech/2023/piyush.kumar23b/data/plantseg'
crop_size = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # Wider scale range for multi-scale robustness
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(4, 21)],  # 256..1280
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # Stronger photometric augmentation for plant images
    dict(type='PhotoMetricDistortion',
         brightness_delta=40,
         contrast_range=(0.5, 1.8),
         saturation_range=(0.4, 1.6),
         hue_delta=25),
    dict(type='RandomFlip', prob=0.5),
    # Random rotation for plant images (plants appear at all angles)
    dict(type='RandomRotate', prob=0.5, degree=30, pad_val=0, seg_pad_val=255),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
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
    num_workers=2,
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

# --------------- Model Settings (ViT-L/14 + Custom Activation v2) ---------------
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth'  # noqa

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
        type='CustomActivationFusionSANCLIPHead',
        num_classes=116,
        san_cfg=dict(
            clip_channels=1024,
            cfg_decoder=dict(num_heads=16),
            # TUNED fusion parameters (gentler initialization)
            init_tau=0.5,      # Was 1.0 -- lower to let model warm up
            init_beta=0.05,    # Was 0.1 -- less aggressive diff bias
            init_alpha=0.3,    # Was 0.5 -- start with weaker fusion
            attn_heads=4,
        ),
        maskgen_cfg=dict(
            num_layers=6,
            embed_dims=1024,
            num_heads=16,
            out_dims=768,
        )))

# --------------- Training Schedule (200k iterations) ---------------
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=200000,
    val_interval=10000,
    val_begin=5000)  # Earlier first validation to catch issues

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

# --------------- Optimizer (matching SAN paper settings) ---------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=4,
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            # Higher LR for fusion params (they are small scalars)
            'tau': dict(lr_mult=5.0, decay_mult=0.),
            'beta': dict(lr_mult=5.0, decay_mult=0.),
            'alpha': dict(lr_mult=5.0, decay_mult=0.),
            # Moderate LR for Q/K/V projections
            'q_proj': dict(lr_mult=2.0),
            'k_proj': dict(lr_mult=2.0),
            'v_proj': dict(lr_mult=2.0),
            'gate_proj': dict(lr_mult=2.0),
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.1, norm_type=2))  # Relaxed from 0.01

# --------------- LR Scheduler (Warmup + Cosine) ---------------
param_scheduler = [
    # Warmup for first 1500 iterations (critical for stability)
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    # Cosine annealing (better than PolyLR for long training)
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=1500,
        end=200000,
        by_epoch=False,
    )
]
