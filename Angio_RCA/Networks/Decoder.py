""" Parts of the U-Net model """

# import torch
# import torch.nn as nn

from .Blocks import *


class StrideResBNDecoderC1(nn.Module):

    def __init__(self, mid_c, n_downsampling, out_kernel, in_kernel):
        super().__init__()

        channel = mid_c * 2 ** n_downsampling

        model = [nn.Conv2d(1, channel, kernel_size=(1, 1), padding=(0, 0)),
                  nn.BatchNorm2d(channel),
                  nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [ResBNUp(mid_c * mult, int(mid_c * mult / 2), k_size=in_kernel, stride=2, pad=in_kernel//2)]

        # original img or mask
        model += [nn.Conv2d(mid_c, 1, kernel_size=out_kernel, padding=out_kernel//2)]
        # original img and mask
        # model += [nn.Conv2d(mid_c, 2, kernel_size=out_kernel, padding=out_kernel // 2)]

        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
