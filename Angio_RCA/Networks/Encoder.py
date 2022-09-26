""" Parts of the Encoder model """
# import torch
# import torch.nn as nn

from .Blocks import *


class StrideResBNEncoderC1(nn.Module):

    def __init__(self, mid_c, n_downsampling, out_kernel, in_kernel):
        super().__init__()

        model = [nn.Conv2d(1, mid_c, kernel_size=out_kernel, padding=out_kernel//2),
                 nn.BatchNorm2d(mid_c),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [ResBNDown(mid_c * mult, mid_c * mult * 2, k_size=in_kernel, stride=2, pad=in_kernel//2)]

        model += [nn.Conv2d(mid_c * 2 ** n_downsampling, 16, kernel_size=(3, 3), padding=(1, 1)),
                  nn.BatchNorm2d(16),
                  nn.ReLU(True),
                  nn.Conv2d(16, 1, kernel_size=(1, 1), padding=(0, 0)),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
