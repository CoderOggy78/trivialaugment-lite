import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.equal_io = in_planes == out_planes
        self.shortcut = None if self.equal_io else nn.Conv2d(
            in_planes, out_planes, 1, stride, 0, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.equal_io else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(int(num_layers)):
            layers.append(BasicBlock(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1,
                drop_rate,
            ))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, drop_rate=0.0):
        super().__init__()
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6

        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, widths[0], widths[1], 1, drop_rate)
        self.block2 = NetworkBlock(n, widths[1], widths[2], 2, drop_rate)
        self.block3 = NetworkBlock(n, widths[2], widths[3], 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(widths[3], num_classes)
        self.num_channels = widths[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.num_channels)
        return self.fc(out)


def wrn_28_10(num_classes=10):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes)


def wrn_40_2(num_classes=10):
    return WideResNet(depth=40, widen_factor=2, num_classes=num_classes)
