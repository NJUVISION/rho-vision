DATASET = 'bdd100k'
CAMERA = 'oneplus'
_base_ = [
    './runtime.py',
    '../_base_/datasets/{}_rgb2{}_raw_512x512.py'.format(DATASET, CAMERA),
]

exp_name = 'unpaired_cycler2r_{}_rgb2{}_raw'.format(DATASET, CAMERA)
work_dir = f'./work_dirs/experiments/{exp_name}'
