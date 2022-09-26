import torch
import torch.nn as nn
import torch.nn.functional as F


# from torchsummary import summary


class SimpleEncoder(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super().__init__()

        self.dropout = nn.Dropout(0.50)
        """
        self.enc_conv1 = nn.Conv2d(1, 16, 11, padding='same')
        self.enc_conv2 = nn.Conv2d(16, 48, 5, padding='same')
        self.enc_conv3 = nn.Conv2d(48, 96, 3, padding='same')
        self.enc_conv4 = nn.Conv2d(96, 128, 2, padding='same')

        self.dec_conv4 = nn.Conv2d(128, 96, 2, padding='same')
        self.dec_conv3 = nn.Conv2d(96, 48, 3, padding='same')
        self.dec_conv2 = nn.Conv2d(48, 16, 5, padding='same')
        self.dec_conv1 = nn.Conv2d(16, 1, 11, padding='same')
        """
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, dilation=1, padding='same')
        self.enc_conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, dilation=1, padding='same')
        self.enc_conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, dilation=1, padding='same')
        self.enc_conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, dilation=1, padding='same')
        self.enc_conv5 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, dilation=1, padding='same')
        self.enc_conv6 = nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=3, stride=1, dilation=1, padding='same')

        self.dec_conv6 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, dilation=1, padding='same')
        self.dec_conv5 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, dilation=1, padding='same')
        self.dec_conv4 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, dilation=1, padding='same')
        self.dec_conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, dilation=1, padding='same')
        self.dec_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, dilation=1, padding='same')
        self.dec_conv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, dilation=1, padding='same')

        self.pool2d = nn.MaxPool2d(2, return_indices=True)
        self.unpool2d = nn.MaxUnpool2d(2)

        # self.bn2d_16 = nn.BatchNorm2d(num_features=16)
        # self.bn2d_48 = nn.BatchNorm2d(num_features=48)
        # self.bn2d_96 = nn.BatchNorm2d(num_features=96)
        # self.bn2d_128 = nn.BatchNorm2d(num_features=128)

        # self.enc_fc1 = nn.Linear(1536, latent_space_dimension)
        # self.enc_fc2 = nn.Linear(255, 1536)
        # self.dec_fc2 = nn.Linear(1536, 255)
        # self.dec_fc1 = nn.Linear(latent_space_dimension, 1536)

        self.bn_8 = nn.BatchNorm2d(num_features=8, affine=False)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')

            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, indices_1 = self.pool2d(torch.relu(self.enc_conv1(x)))
        x, indices_2 = self.pool2d(torch.relu(self.enc_conv2(x)))
        x, indices_3 = self.pool2d(torch.relu(self.enc_conv3(x)))
        x, indices_4 = self.pool2d(torch.relu(self.enc_conv4(x)))
        x, indices_5 = self.pool2d(torch.relu(self.enc_conv5(x)))
        x, indices_6 = self.pool2d(torch.relu(self.enc_conv6(x)))

        # x= x.view(-1,128*2*2)
        # print(x.shape)
        # x = torch.relu(self.enc_fc1(x))
        # x = torch.relu(self.enc_fc2(x))

        # latent = x.view(-1, 8, 8 * 8)
        latent = x

        # x = torch.relu(self.dec_fc2(x))
        # x = torch.relu(self.dec_fc1(x))
        # x = x.view(-1,96,4,4)
        x = self.unpool2d(x, indices_6)
        x = torch.relu(self.dec_conv6(x))
        layer_1=x
        x = self.unpool2d(x, indices_5)
        x = torch.relu(self.dec_conv5(x))
        layer_2 = x
        x = self.unpool2d(x, indices_4)
        x = torch.relu(self.dec_conv4(x))

        x = self.unpool2d(x, indices_3)
        x = torch.relu(self.dec_conv3(x))

        x = self.unpool2d(x, indices_2)
        x = torch.relu(self.dec_conv2(x))

        x = self.unpool2d(x, indices_1)
        x = torch.relu(self.dec_conv1(x))

        return latent, layer_1, layer_2, x
