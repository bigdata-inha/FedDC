import torch
import torch.nn as nn
import torch.nn.functional as F
from group_normalization import GroupNorm2d
from mabn import MABN2d

def norm2d(planes, group_norm):
    if group_norm=='group_norm':
        return nn.GroupNorm(2, planes)
    if group_norm=='instancenorm':
        return nn.GroupNorm(planes, planes)
    elif group_norm=='mabn':
        return MABN2d(planes, B=20, real_B=100, momentum=0.997, eps=1e-5)
    else:
        return nn.BatchNorm2d(planes,eps=1e-5, affine=True, track_running_stats=True)


def conv2d(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_stand=0):
    if weight_stand==1:
        return Conv2d_stand(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
    elif weight_stand==2:
        return CenConv2d(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

class CenConv2d(nn.Conv2d):
    """Conv2d layer with Weight Centralization. 
    The args is exactly same as torch.nn.Conv2d. It's suggested to set bias=False when
    using CenConv2d with MABN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CenConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                padding, dilation, groups, bias) 

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        return F.conv2d(x, weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

class Conv2d_stand(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_stand, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm='group_norm', weight_stand=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_stand=weight_stand)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,  weight_stand=weight_stand)
        self.bn2 = norm2d(planes, group_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, weight_stand=weight_stand),
                norm2d(self.expansion * planes, group_norm)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_nobn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm='group_norm', weight_stand=0):
        super(BasicBlock_nobn, self).__init__()
        self.conv1 = conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_stand=weight_stand)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,  weight_stand=weight_stand)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, weight_stand=weight_stand),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_noshortcut(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_noshortcut(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group_norm='group_norm', weight_stand=0):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1 = conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, weight_stand=weight_stand)
        self.bn1 = norm2d(16, group_norm)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, group_norm=group_norm, weight_stand=weight_stand)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride, group_norm='group_norm', weight_stand=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, group_norm, weight_stand))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        # out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_cifar_nobn(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group_norm='group_norm', weight_stand=0):
        super(ResNet_cifar_nobn, self).__init__()
        self.in_planes = 16

        self.conv1 = conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, weight_stand=weight_stand)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, group_norm=group_norm, weight_stand=weight_stand)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride, group_norm='group_norm', weight_stand=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, group_norm, weight_stand))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        # out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class WResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(WResNet_cifar, self).__init__()
        self.in_planes = 16 * k

        self.conv1 = nn.Conv2d(3, 16 * k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * k)
        self.layer1 = self._make_layer(block, 16 * k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * k, num_blocks[2], stride=2)
        self.fc = nn.Linear(64 * k * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ImageNet models
def ResNet18(num_classes=10, group_norm='group_norm', weight_stand=0):
    if weight_stand==0 and group_norm=='mabn':
        weight_stand=2
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, group_norm=group_norm, weight_stand=weight_stand)


def ResNet18_noshort():
    return ResNet(BasicBlock_noshortcut, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet34_noshort():
    return ResNet(BasicBlock_noshortcut, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet50_noshort():
    return ResNet(Bottleneck_noshortcut, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet101_noshort():
    return ResNet(Bottleneck_noshortcut, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def ResNet152_noshort():
    return ResNet(Bottleneck_noshortcut, [3, 8, 36, 3])


# CIFAR-10 models
def ResNet20():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n, n, n])


def ResNet20_noshort():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n])


def ResNet32_test(num_classes=10,  group_norm='group_norm', weight_stand=0):
    if weight_stand==0 and group_norm=='mabn':
        weight_stand=2
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n, n, n],group_norm=group_norm, weight_stand=weight_stand, num_classes=num_classes)

def ResNet32_nobn(num_classes=10,  group_norm='group_norm', weight_stand=0):
    if weight_stand==0 and group_norm=='mabn':
        weight_stand=2
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar_nobn(BasicBlock_nobn, [n, n, n],group_norm=group_norm, weight_stand=weight_stand, num_classes=num_classes)


def ResNet32_noshort():
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n])


def ResNet44_noshort():
    depth = 44
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n])


def ResNet50_16_noshort():
    depth = 50
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n])


def ResNet56():
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n, n, n])


def ResNet56_noshort():
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n])


def ResNet110():
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n, n, n])


def ResNet110_noshort():
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n])


def WRN56_2():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n, n, n], 2)


def WRN56_4():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n, n, n], 4)


def WRN56_8():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n, n, n], 8)


def WRN56_2_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 2)


def WRN56_4_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 4)


def WRN56_8_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 8)


def WRN110_2_noshort():
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 2)


def WRN110_4_noshort():
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 4)
