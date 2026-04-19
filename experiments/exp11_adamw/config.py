# ============================================================
# Experiment: AdamW optimizer baseline
# ============================================================

_base_ = ['../../configs/san/san-vit-l14_plantseg115_server.py']

optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.0),
            'cls_token': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
