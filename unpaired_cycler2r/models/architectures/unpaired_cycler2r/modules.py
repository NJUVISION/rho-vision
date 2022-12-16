import torch
import torch.nn as nn


class TinyEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channnels=64, input_size=128):
        super().__init__()
        self.input_size = input_size
        self.net = torch.nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 1, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, channnels, 3, 1, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(channnels, channnels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.linear = nn.Sequential(
            nn.Linear(channnels, 256),
            nn.Linear(256, out_channels)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def differentiable_histogram(x, bins=255, min=0.0, max=1.0):
    if len(x.shape) == 4:
        n_samples, n_chns, _, _ = x.shape
    elif len(x.shape) == 2:
        n_samples, n_chns = 1, 1
    else:
        raise AssertionError('The dimension of input tensor should be 2 or 4.')

    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    delta = (max - min) / bins

    BIN_Table = torch.arange(start=0, end=bins + 1, step=1) * delta

    for dim in range(1, bins - 1, 1):
        h_r = BIN_Table[dim].item()  # h_r
        h_r_sub_1 = BIN_Table[dim - 1].item()  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
        hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)

    return hist_torch / delta


def differentiable_uv_histogram(x, bins=32):
    b, c, _, _ = x.shape
    assert c == 3
    hist_uv_map = torch.zeros(b, 1, bins, bins, device=x.device)
    x = torch.clamp(x, 1e-6, 1)
    u, v = torch.log(x[:, 1] / x[:, 0]), torch.log(x[:, 1] / x[:, 2])  # [b, h, w]
    y = torch.sqrt(torch.pow(x, 2).sum(1))  # [b, h, w]
    y, u, v = y.view([b, -1]), u.view([b, -1]), v.view([b, -1])
    eta = 0.025 * 256 / bins

    for u_c in range(1, bins + 1):
        for v_c in range(1, bins + 1):
            u_sub = (u_c - 0.5) * eta
            u_plus = (u_c + 0.5) * eta
            v_sub = (v_c - 0.5) * eta
            v_plus = (v_c + 0.5) * eta
            u_mask_sub = ((u_sub <= u) & (u < u_c)).float().detach()
            v_mask_sub = ((v_sub <= v) & (v < v_c)).float().detach()
            u_mask_plus = ((u_c <= u) & (u < u_plus)).float().detach()
            v_mask_plus = ((v_c <= v) & (v < v_plus)).float().detach()
            hist_uv_map[:, 0, u_c - 1, v_c - 1] += torch.sum(y * (u - u_sub) * u_mask_sub, -1)
            hist_uv_map[:, 0, u_c - 1, v_c - 1] += torch.sum(y * (v - v_sub) * v_mask_sub, -1)
            hist_uv_map[:, 0, u_c - 1, v_c - 1] += torch.sum(y * (u - u_plus) * u_mask_plus, -1)
            hist_uv_map[:, 0, u_c - 1, v_c - 1] += torch.sum(y * (v - v_plus) * v_mask_plus, -1)
    hist_uv_map = hist_uv_map/hist_uv_map.view([b, -1]).sum(-1).view([b, 1, 1, 1])
    hist_uv_map = torch.sqrt(hist_uv_map)
    return hist_uv_map
