_domain_a = 'raw'  # set by user
_domain_b = 'rgb'  # set by user
model = dict(
    type='UnpairedCycleR2R',
    lambda_bright_diversity=30.0,
    lambda_color_diversity=3.0,
    generator=dict(
        type='inverseISP'),
    discriminator=dict(
        type='HistAwareDiscriminator'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    default_domain=_domain_b,
    reachable_domains=[_domain_a, _domain_b],
    related_domains=[_domain_a, _domain_b],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{_domain_b}',
                target=f'real_{_domain_b}',
            ),
            reduction='mean')
    ])
train_cfg = dict(buffer_size=50)
test_cfg = None
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=[f'real_{_domain_a}', f'fake_{_domain_b}', f'cycle_{_domain_a}', f'real_{_domain_b}',
                       f'fake_{_domain_a}', f'fake_{_domain_a}_diversity', f'cycle_{_domain_b}'],
        rerange=False,
        bgr2rgb=False,
        interval=200)
]
