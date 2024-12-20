import torch.nn as nn
import torch
from collections import OrderedDict


class Dcnn_Pytorch(nn.Module):
    def __init__(self, init_weights=False, mode='normal', stage='DM'):
        super(Dcnn_Pytorch, self).__init__()
        self.stage = stage
        if self.stage != 'DM': # denoising stage, need noise layer
            self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = BasicConv2d(32, 64, kernel_size=3, padding=1)

        layers = OrderedDict([])

        if self.stage != 'DM':
            self.dm = BasicConv2d(128, 64, kernel_size=3, padding=1)

        for i in range(16):
            if mode == 'normal':
                layers["Inception{}".format(i + 1)] = Inception(64, 32, 16, 32)
            else:
                layers["Inception{}".format(i + 1)] = Inception_lightweight(64, 16, 16, 32)

        self.main_processor = nn.Sequential(layers)
        self.layer2 = BasicConv2d(64, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, skip, noise):
        if self.stage != 'DM':
            y = torch.cat([x, noise], dim=1)
            y = self.conv1(y)
        else:
            y = self.conv1(x)

        y = self.relu1(y)
        y = self.layer1(y)

        if self.stage != 'DM':
            y = torch.cat([y, skip], dim=1)
            y = self.dm(y)

        y = self.main_processor(y)

        if self.stage == 'DM':  # demosaic stage, output features
            z = y

        y = self.layer2(y)
        y = self.conv2(y)

        if self.stage == 'DM':
            return x + y, z
        else:
            return x - y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Inception(nn.Module):
    def __init__(self, in_channels=64, ch1=32, ch3_1=16, ch3_2=32):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch1, kernel_size=1),
            BasicConv2d(ch1, ch3_1, kernel_size=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch1, kernel_size=1),
            BasicConv2d(ch1, ch1, kernel_size=3, padding=1),
            BasicConv2d(ch1, ch3_1, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch1, kernel_size=1),
            BasicConv2d(ch1, ch3_2, kernel_size=3, padding=1),
            BasicConv2d(ch3_2, ch3_2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        output = [branch1, branch2, branch3]

        return torch.cat(output, 1) + x


class Inception_lightweight(nn.Module):
    def __init__(self, in_channels=64, ch1=16, ch3_1=16, ch3_2=32):
        super(Inception_lightweight, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch1, kernel_size=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch1, kernel_size=1),
            BasicConv2d(ch1, ch3_1, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch1, kernel_size=1),
            BasicConv2d(ch1, ch3_2, kernel_size=3, padding=1),
            BasicConv2d(ch3_2, ch3_2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        output = [branch1, branch2, branch3]

        return torch.cat(output, 1) + x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
