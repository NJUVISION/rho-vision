import argparse
import numpy as np
from collections import OrderedDict
from imageio import imread, imwrite

import torch
from torch import nn
import torch.distributions as D

from mmgen.models import build_module
from unpaired_cycler2r.models import *


class DemoModel(nn.Module):

    def __init__(self, ckpt_path) -> None:
        super().__init__()
        # load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('generator.'):
                state_dict[k[len('generator.'):]] = v

        # get invISP model
        self.model = build_module(dict(type='inverseISP'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _get_illumination_condition(self, img):
        mean_var = self.model.color_condition_gen(img)
        m = D.Normal(mean_var[:, 0],
                     torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        color_condition = m.sample()
        mean_var = self.model.bright_condition_gen(img)
        m = D.Normal(mean_var[:, 0],
                     torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        bright_condition = m.sample()
        condition = torch.cat(
            [color_condition[:, None], bright_condition[:, None]], 1)
        return condition

    def _mosaic(self, x):
        h, w = x.shape[2:]
        _x = torch.zeros(x.shape[0], 4, h // 2, w // 2, device=x.device)
        _x[:, 0] = x[:, 0, 0::2, 0::2]
        _x[:, 1] = x[:, 0, 0::2, 1::2]
        _x[:, 2] = x[:, 0, 1::2, 0::2]
        _x[:, 3] = x[:, 0, 1::2, 1::2]
        return _x

    def forward(self, rgb, mosaic=False):
        with torch.no_grad():
            # get illumination condition
            condition = self._get_illumination_condition(rgb)
            # get simulated RAW image
            raw = self.model(rgb, condition, rev=False)
            raw = torch.clamp(raw, 0, 1)
            if mosaic:
                raw = self._mosaic(raw)
        return raw


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--ckpt', type=str)
    args.add_argument('--rgb', type=str)
    args = args.parse_args()

    model = DemoModel(args.ckpt)
    img = imread(args.rgb).astype(np.float32) / 255
    img = torch.from_numpy(img).permute(2, 0, 1)[None]

    model = model.cuda()
    img = img.cuda()

    x = model(img, mosaic=False)
    x = x[0].permute(1, 2, 0).cpu().numpy()
    x = (x * 255).astype(np.uint8)
    imwrite('simulated_preview.png', x)
