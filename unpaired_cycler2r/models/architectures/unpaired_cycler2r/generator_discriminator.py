from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
from mmcv.runner import load_checkpoint

from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger
from .modules import TinyEncoder, differentiable_histogram


@MODULES.register_module()
class inverseISP(nn.Module):
    def __init__(self, wb=None, ccm=None, global_size=128):
        super().__init__()
        wb = [
            [2.0931, 1.6701],
            [2.1932, 1.7702],
            [2.2933, 1.8703],
            [2.3934, 1.9704],
            [2.4935, 1.9705]
        ] if wb is None else wb
        self.wb = nn.Parameter(torch.FloatTensor(wb), requires_grad=True)

        ccm = [
            [[1.67557, -0.52636, -0.04920],
             [-0.16799, 1.32824, -0.36024],
             [0.03188, -0.22302, 1.59114]],
            [[1.57558, -0.52637, -0.04921],
             [-0.16798, 1.52823, -0.36023],
             [0.031885, -0.42303, 1.39115]]
        ] if ccm is None else ccm
        self.ccm = nn.Parameter(torch.FloatTensor(ccm), requires_grad=True)

        self.resize_fn = partial(nn.functional.interpolate, size=global_size)
        self.global_size = global_size

        self.color_condition_gen = TinyEncoder(3, 2)
        self.bright_condition_gen = TinyEncoder(3, 2)

        # inverse ISP
        self.ccm_estimator = TinyEncoder(7, 1)
        self.bright_estimator = TinyEncoder(3, 1)
        self.wb_estimator = TinyEncoder(7, 1)

        # ISP
        self.wb_evaluator = TinyEncoder(6, 1)
        self.bright_evaluator = TinyEncoder(2, 1)
        self.ccm_evaluator = TinyEncoder(6, 1)

        self.counter = 0

    def initialize(self):
        pass

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

    def safe_inverse_gain(self, x, gain):
        gray = x.mean(1, keepdim=True)
        inflection = 0.9
        mask = torch.maximum(gray - inflection, torch.zeros_like(gray)) / (1 - inflection)
        mask = mask ** 2
        mask = torch.clamp(mask, 0, 1)
        safe_gain = torch.maximum(mask + (1 - mask) * gain, gain)
        x = x * safe_gain
        return x

    def rgb2raw(self, x, condition):
        b = x.shape[0]
        gs = self.global_size
        ccm_condition = torch.ones([b, self.ccm.shape[0], 1, gs, gs], device=x.device)
        ccm_condition *= condition[:, 0].view([b, 1, 1, 1, 1])
        ccm_condition = ccm_condition.view([-1, 1, gs, gs])

        b_condition = torch.ones([b, 1, gs, gs], device=x.device)
        b_condition *= condition[:, 1].view([b, 1, 1, 1])
        
        wb_condition = torch.ones([b, self.wb.shape[0], 1, gs, gs], device=x.device)
        wb_condition *= condition[:, 0].view([b, 1, 1, 1, 1])
        wb_condition = wb_condition.view([-1, 1, gs, gs])

        ## inverse gamma
        x = torch.maximum(x, 1e-8 * torch.ones_like(x))
        x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

        ## inverse ccm
        x_resize = self.resize_fn(x)
        ccm = self.ccm / self.ccm.reshape([-1, 3]).sum(1).reshape([-1, 3, 1])
        inv_ccm = torch.linalg.pinv(ccm.transpose(-2, -1))
        ccm_preview = torch.einsum('bchw,ncj->bnjhw', x_resize, inv_ccm)
        ccm_preview = ccm_preview.reshape([-1, 3, gs, gs])
        ccm_preview = torch.cat([ccm_preview, ccm_preview**2, ccm_condition], 1)
        ccm_prob = self.ccm_estimator(ccm_preview).view([b, -1])
        ccm_prob = torch.softmax(ccm_prob, 1)
        ccm = ccm[None] * ccm_prob.view([b, -1, 1, 1]) # [B, N, 3, 3]
        ccm = ccm.sum(1)  # [B, 3, 3]
        inv_ccm = torch.linalg.pinv(ccm.transpose(-2, -1))  # [B, 3, 3]
        x = torch.einsum('bchw,bcj->bjhw', [x, inv_ccm])  # [B, 3, H, W]

        ## inverse brightness adjustment
        x_resize = self.resize_fn(x)
        x_resize = torch.mean(x_resize, 1, keepdim=True)
        x_resize = torch.cat([x_resize, x_resize**2, b_condition], 1)
        bright = torch.tanh(self.bright_estimator(x_resize))
        bright_adjust = bright * 0.8 + 0.8
        x = self.safe_inverse_gain(x, bright_adjust.view([b, 1, 1, 1]))  # [B, 3, H, W]
        
        ## inverse awb
        x_resize = self.resize_fn(x)
        gain = torch.ones([b, self.wb.shape[0], 3]).to(x.device)
        gain[..., (0, 2)] = 1 / self.wb[None]  # [B, N, 3]
        wb_preview = x_resize[:, None].repeat([1, self.wb.shape[0], 1, 1, 1])  # [B, N, 3, 128, 128]
        wb_preview = wb_preview.reshape([-1, 3, gs, gs])  # [B*N, 3, 128, 128]
        gain = gain.reshape([-1, 3, 1, 1])  # [B*N, 3, 1, 1]
        wb_preview = self.safe_inverse_gain(wb_preview, gain)
        wb_preview = torch.cat([wb_preview, wb_preview**2, wb_condition], 1)
        
        wb_prob = self.wb_estimator(wb_preview).view([b, -1, 1, 1, 1])
        wb_prob = torch.softmax(wb_prob, 1)
        gain = gain.view([b, -1, 3, 1, 1]) * wb_prob  # [B, N, 3, 1, 1]
        gain = gain.sum(1)  # [B, 3, 1, 1]
        x = self.safe_inverse_gain(x, gain)  # [B, 3, H, W]

        # mosaic
        # x = self.mosaic(x)
        return x

    def raw2rgb(self, x):
        # global
        b = x.shape[0]
        gs = self.global_size

        ## demosaic
        # x = self.demosaic(x)

        ## white balance
        x_resize = self.resize_fn(x)
        x_resize = x_resize[:, None].repeat([1, self.wb.shape[0], 1, 1, 1])  # [B, N, 3, 128, 128]
        x_resize[:, :, (0, 2)] = x_resize[:, :, (0, 2)] * self.wb.view([1, -1, 2, 1, 1])  # [B, N, 3, 128, 128]
        x_resize = x_resize.reshape([-1, 3, gs, gs])  # [B*N, 3, 128, 128]
        x_resize = torch.cat([x_resize, x_resize**2], 1)
        wb_prob = self.wb_evaluator(x_resize).reshape([b, self.wb.shape[0]])  # [B, N]
        wb_prob = torch.softmax(wb_prob, 1)
        wb_prob = wb_prob.view([b, -1, 1, 1, 1])  # [B, N, 1, 1, 1]
        wb = self.wb.view([1, -1, 2, 1, 1])  # [B, N, 2, 1, 1]
        wb = (wb * wb_prob).sum(1)  # [B, 2, 1, 1]
        x[:, (0, 2)] = x[:, (0, 2)] * wb

        ## brightness adjustment
        x_resize = self.resize_fn(x)
        x_resize = torch.mean(x_resize, 1, keepdim=True)
        x_resize = torch.cat([x_resize, x_resize**2], 1)
        bright = torch.tanh(self.bright_evaluator(x_resize))
        bright_adjust = 0.8 + bright * 0.8
        bright_adjust = torch.abs(bright_adjust)
        x = x / bright_adjust[:, :, None, None]

        ## ccm
        x_resize = self.resize_fn(x)
        ccm = self.ccm / self.ccm.sum(2, keepdims=True)
        ccm = ccm.transpose(1, 2)
        ccm_preview = torch.einsum('bchw,ncj->bnjhw', x_resize, ccm)  # [B, N, 3, 128, 128]
        ccm_preview = ccm_preview.reshape([-1, 3, gs, gs])  # [B*N, 3, 128, 128]
        ccm_preview = torch.cat([ccm_preview, ccm_preview**2], 1)
        ccm_prob = self.ccm_evaluator(ccm_preview).reshape([b, self.ccm.shape[0]])  # [B, N]
        ccm_prob = torch.softmax(ccm_prob, 1)
        ccm_prob = ccm_prob.view([b, -1, 1, 1])  # [B, N, 1, 1]
        ccm = ccm[None] * ccm_prob  # [B, N, 3, 3]
        ccm = ccm.sum(1)  # [B, 3, 3]
        x = torch.einsum('bchw,bcj->bjhw', x, ccm)  # [B, 3, H, W]

        ## gamma correction
        x = torch.maximum(x, 1e-8 * torch.ones_like(x))
        x = torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1 / 2.4) - 0.055)
        return x

    def forward(self, x, condition=None, rev=False):
        x = x.clone()
        # inverse ISP
        if not rev:
            x = self.rgb2raw(x, condition)
            log_dict = {}
            log_dict['wb'] = self.wb.detach().cpu()
            log_dict['ccm'] = self.ccm.detach().cpu()
            # log
            if self.training and (not torch.distributed.is_initialized() or dist.get_rank() == 0):
                self.counter += 1
                if self.counter == 200:
                    self.counter = 0
                    for k, v in log_dict.items():
                        print(k, ':', '\n', v)
        # ISP
        else:
            x = self.raw2rgb(x)

        return x


class FCDiscriminator(nn.Module):

    def __init__(self, in_channels=3, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = x.reshape([x.shape[0], -1]).mean(1)
        return x


@MODULES.register_module()
class HistAwareDiscriminator(nn.Module):

    def __init__(self, in_channels=3, ndf=64, bins=255, global_size=128):
        super(HistAwareDiscriminator, self).__init__()
        self.bins = bins
        self.bright_fcd = nn.Sequential(
            nn.Linear(bins * in_channels, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Linear(256, 1)
        )
        self.local_fcd = FCDiscriminator(in_channels=in_channels, ndf=ndf)
        self.resize_fn = partial(nn.functional.interpolate, size=(global_size, global_size))
        self.uv_fcd = FCDiscriminator(in_channels=2, ndf=ndf)

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= 1.  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= 1.
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

    def forward(self, x):
        local_judge = self.local_fcd(x.clone())

        hist = differentiable_histogram(x, self.bins)  # [B, C, 256]
        hist /= x.shape[2] * x.shape[3]
        bright_judge = self.bright_fcd(hist.reshape([hist.shape[0], -1]))

        x = self.resize_fn(torch.clamp(x.clone(), 1e-6, 1))
        u, v = torch.log(x[:, 1] / x[:, 0]), torch.log(x[:, 1] / x[:, 2])  # [b, h, w]
        uv_judge = self.uv_fcd(torch.cat([u[:, None], v[:, None]], 1))

        combine_judge = local_judge + bright_judge + uv_judge
        return combine_judge
