_base_ = [
    '../_base_/models/unpaired_cycler2r/unpaired_cycler2r.py',
]
# log
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

# dist
# use dynamic runner
runner = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
find_unused_parameters = False
cudnn_benchmark = True
use_ddp_wrapper = True
total_iters = 20000
workflow = [('train', 1)]

# learning policy
optimizer = dict(
    generator=dict(type='AdamW', lr=5e-4),
    discriminators=dict(type='Adam', lr=5e-5, betas=(0.5, 0.999)))

lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=10000, interval=400)

# evalutation
num_images = 20
evaluation = dict(
    type='TranslationEvalHook',
    target_domain='raw',
    interval=10000,
    metrics=[
        dict(type='FID', num_images=num_images, bgr2rgb=False),
        dict(
            type='IS',
            num_images=num_images,
            image_shape=(3, 256, 256),
            inception_args=dict(type='pytorch'))
    ],
    best_metric=['fid', 'is'])

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
