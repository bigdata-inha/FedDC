'''Pre-activation ResNet in PyTorch.
ResNet v2

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import math
from random import weibullvariate
import torch
import torch.nn as nn
import torch.nn.functional as F
from mabn import MABN2d
from batchrenorm import BatchRenorm2d
from evonorm import EvoNorm2D
from torch.nn.parameter import Parameter

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

def norm2d(planes, group_norm):
    if group_norm=='group_norm':
        return nn.GroupNorm(2, planes)
    elif group_norm=='mabn':
        return MABN2d(planes, B=20, real_B=100, momentum=0.997, eps=1e-5)
    elif group_norm=='batch_renorm':
        return BatchRenorm2d(planes)
    elif group_norm=='evonorm':
        return EvoNorm2D(planes)
    elif group_norm=='instancenorm':
        return nn.GroupNorm(planes, planes)
    else:
        return nn.BatchNorm2d(planes)

class ZeroMean_Classifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(ZeroMean_Classifier, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)    
    def forward(self, input):
        weight = self.weight
        weight_mean = weight.mean(dim=1,keepdim=True)
        weight = weight - weight_mean
        return F.linear(input,weight, self.bias)


class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())



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

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm='group_norm', weight_stand=0, avoid_norm=False):
        super(PreActBlock, self).__init__()
        if not avoid_norm:
            self.bn1 = norm2d(in_planes, group_norm)
        self.avoid_norm = avoid_norm
        self.conv1 = conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_stand=weight_stand)
        self.bn2 = norm2d(planes, group_norm)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_stand=weight_stand)

            
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, weight_stand=weight_stand),
                norm2d(self.expansion*planes, group_norm=group_norm)
            )
       
    def forward(self, x):
        if self.avoid_norm:
            out = self.conv1(x)
            shortcut = x
        else:
            out = self.bn1(x)
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
        out = self.conv2(self.bn2(out))
        out += shortcut
        return out

class PreActBlock_nobn(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm='group_norm', weight_stand=0, avoid_norm=False):
        super(PreActBlock_nobn, self).__init__()
  
        self.conv1 = conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_stand=weight_stand)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_stand=weight_stand)

            
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, weight_stand=weight_stand),
            )
       
    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_norm='group_norm', weight_stand=0):
        super(PreActBottleneck, self).__init__()
        self.bn1 = norm2d(in_planes, group_norm)
        self.conv1 = conv2d(in_planes, planes, kernel_size=1, bias=False, weight_stand=weight_stand)
        self.bn2 = norm2d(planes, group_norm)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_stand=weight_stand)
        self.bn3 = norm2d(planes, group_norm)
        self.conv3 = conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, weight_stand=weight_stand)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, weight_stand=weight_stand)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, group_norm='group_norm', weight_stand=0):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, weight_stand=weight_stand)
        self.bn1 = norm2d(64, group_norm)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, group_norm=group_norm, is_first_layer=True, weight_stand=weight_stand)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.bn2 = norm2d(512, group_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ZeroMean_Classifier(512*block.expansion, num_classes)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, num_blocks, stride, group_norm='group_norm', weight_stand=0, is_first_layer=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, group_norm=group_norm, weight_stand=weight_stand, avoid_norm=is_first_layer and (i==0)))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn2(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        out = self.classifier(features)
        return out


class PreActResNet_nobn(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, group_norm='group_norm', weight_stand=0):
        super(PreActResNet_nobn, self).__init__()
        self.in_planes = 64

        self.conv1 = conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, weight_stand=weight_stand)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, group_norm=group_norm, is_first_layer=True, weight_stand=weight_stand)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(512*block.expansion, num_classes)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, num_blocks, stride, group_norm='group_norm', weight_stand=0, is_first_layer=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, group_norm=group_norm, weight_stand=weight_stand, avoid_norm=is_first_layer and (i==0)))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class PreActResNet_nonClassifier(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, group_norm='group_norm', weight_stand=0):
        super(PreActResNet_nonClassifier, self).__init__()
        self.in_planes = 64
        
        self.conv1 = conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, weight_stand=weight_stand)
        self.bn1 = norm2d(64, group_norm)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, group_norm=group_norm, is_first_layer=True, weight_stand=weight_stand)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, group_norm=group_norm, weight_stand=weight_stand)
        self.bn2 = norm2d(512, group_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, num_blocks, stride, group_norm='group_norm', weight_stand=0, is_first_layer=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, group_norm=group_norm, weight_stand=weight_stand, avoid_norm=is_first_layer and (i==0)))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn2(out)
        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        return features


def PreActResNet18(num_classes=100, group_norm='group_norm', weight_stand=0):
    if weight_stand==0 and group_norm=='mabn':
        weight_stand=2
    return PreActResNet(PreActBlock, [2,2,2,2],num_classes=num_classes, group_norm=group_norm, weight_stand=weight_stand)

def PreActResNet18_nobn(num_classes=100, group_norm='group_norm', weight_stand=0):
    if weight_stand==0 and group_norm=='mabn':
        weight_stand=2
    return PreActResNet_nobn(PreActBlock_nobn, [2,2,2,2], num_classes=num_classes, group_norm=group_norm, weight_stand=weight_stand)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])

def PreActResNet18_nonClassifier(num_classes=10, group_norm='group_norm', weight_stand=0):
    if weight_stand==0 and group_norm=='mabn':
        weight_stand=2
    return PreActResNet_nonClassifier(PreActBlock, [2,2,2,2],num_classes=num_classes, group_norm=group_norm, weight_stand=weight_stand)

model_dict = {
    'resnet18': [PreActResNet18_nonClassifier, 512],
    'resnet34': [PreActResNet34, 512],
    'resnet50': [PreActResNet50, 2048],
    'resnet101': [PreActResNet101, 2048],
}
class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128, num_classes=10):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        # classifier
        self.classifier =  nn.Linear(dim_in, num_classes)
    def forward(self, x):
        feat = self.encoder(x)
        norm_feat = F.normalize(self.head(feat), dim=1)
        output = self.classifier(feat)
        return output, norm_feat 

def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
