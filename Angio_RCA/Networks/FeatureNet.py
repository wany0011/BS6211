""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Blocks import *


class FeatureNetDynT(nn.Module):

    def __init__(self, input_feature, neuron_list_b4_t, neuron_list_after_t, n_pts):
        super().__init__()
        self.input_feature = input_feature
        self.n_b4_T = neuron_list_b4_t
        self.n_after_T = neuron_list_after_t
        self.n_pts = n_pts

        model_b4_t = [nn.Flatten(1, -1)]
        pre_neuron = self.input_feature

        for n_neuron in neuron_list_b4_t:
            model_b4_t += [nn.Linear(pre_neuron, n_neuron)]
            model_b4_t += [nn.ReLU(True)]
            pre_neuron = n_neuron

        self.mid_feature = pre_neuron
        self.model_b4_t = nn.Sequential(*model_b4_t)

        model_after_t = []
        pre_neuron += 1
        for n_neuron in neuron_list_after_t:
            model_after_t += [nn.Linear(pre_neuron, n_neuron)]
            model_after_t += [nn.ReLU(True)]
            pre_neuron = n_neuron

        model_after_t += [nn.Linear(pre_neuron, 2)]
        model_after_t += [nn.Sigmoid()]

        self.model_after_t = nn.Sequential(*model_after_t)

    def forward(self, y, t):
        y = self.model_b4_t(y)
        y = y.view(-1, 1, self.mid_feature)
        y = y.repeat(1, self.n_pts, 1)
        t = t.view(-1, self.n_pts, 1)
        # print('!!!!!!!!!!!!!!!', t.shape, y.shape)
        # print(t)
        y = torch.cat([y, t], dim=2)
        # print(y[0, :, -2:])
        # print(y.shape)
        # inputs = (y, t)
        y = self.model_after_t(y)
        # print('Reached here.', y.shape)
        return y
