""" Full assembly of the parts to form the complete network """

from .Encoder import *
from .Decoder import *
from .FeatureNet import *


class FinalModel(nn.Module):
    def __init__(self, n_channel_2d, n_downsampling, out_kernel, in_kernel,
                 neuron_list_b4_t, neuron_list_after_t, n_pts):
        super().__init__()
        self.ip_feature = int((128 / 2 ** n_downsampling) ** 2)
        # print('ip_feature', self.ip_feature)

        self.encoder = StrideResBNEncoderC1(n_channel_2d, n_downsampling, out_kernel, in_kernel)
        self.feature_net = FeatureNetDynT(self.ip_feature, neuron_list_b4_t, neuron_list_after_t, n_pts)

    def forward(self, x, t):
        img = self.encoder(x)
        # print(img.shape)
        coord = self.feature_net(img, t)
        # print('img shape {}, coord shape {}'.format(img.shape, coord.shape))
        return img, coord
