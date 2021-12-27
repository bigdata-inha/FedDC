#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from mabn import MABN2d

from collections import OrderedDict
import numpy as np
import torch
from copy import deepcopy


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

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if args.dataset=='cifar100':
            self.fc1 = nn.Linear(2880, 120)
        else:
            self.fc1 = nn.Linear(16*5*5, 120)
        self.dataset = args.dataset
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

        self.norm = args.norm

        if self.norm == 'batch_norm':
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)
        elif self.norm == 'group_norm':
            self.gn1 = nn.GroupNorm(2, 6)
            self.gn2 = nn.GroupNorm(2, 16)
        elif self.norm == 'instance_norm':
            self.gn1 = nn.GroupNorm(6, 6)
            self.gn2 = nn.GroupNorm(16, 16)

    def forward(self, x):
        if self.norm == 'batch_norm':
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
        elif self.norm == 'group_norm' or self.norm == 'instance_norm':
            x = self.pool(F.relu(self.gn1(self.conv1(x))))
            x = self.pool(F.relu(self.gn2(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
      
        if self.dataset == 'cifar100':
            x = x.view(-1, 2880)
        else:
            x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNCifar_WS(nn.Module):
    def __init__(self, args):
        super(CNNCifar_WS, self).__init__()
        self.conv1 = Conv2d_stand(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2d_stand(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

        self.norm = args.norm

        if self.norm == 'batch_norm':
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)
        elif self.norm == 'group_norm':
            self.gn1 = nn.GroupNorm(2, 6)
            self.gn2 = nn.GroupNorm(2, 16)
        elif self.norm == 'instance_norm':
            self.gn1 = nn.GroupNorm(64, 6)
            self.gn2 = nn.GroupNorm(64, 16)
    def forward(self, x):
        if self.norm == 'batch_norm':
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
        elif self.norm == 'group_norm' or self.norm == 'instance_norm':
            x = self.pool(F.relu(self.gn1(self.conv1(x))))
            x = self.pool(F.relu(self.gn2(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNCifar_fedVC(nn.Module):
    def __init__(self, args, weight_stand=0):
        super(CNNCifar_fedVC, self).__init__()

        if weight_stand:
            self.conv1 = Conv2d_stand(3, 64, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = Conv2d_stand(64, 64, 5)
        elif weight_stand ==0 and args.norm == 'mabn':
            self.conv1 = CenConv2d(3, 64, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = CenConv2d(64, 64, 5)
        else:
            self.conv1 = nn.Conv2d(3, 64, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(64, 64, 5)
            
        self.fc1 = nn.Linear(64 * 5 * 5, 394)
        self.fc2 = nn.Linear(394, 192)
        self.fc3 = nn.Linear(192, args.num_classes)
        
        self.norm = args.norm

        if self.norm == 'batch_norm':
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
        elif self.norm == 'group_norm':
            self.gn1 = nn.GroupNorm(2, 64)
            self.gn2 = nn.GroupNorm(2, 64)
        elif self.norm == 'instance_norm':
            self.gn1 = nn.GroupNorm(64, 64)
            self.gn2 = nn.GroupNorm(64, 64)
        elif self.norm == 'mabn':
            self.bn1 = MABN2d(64)
            self.bn2 = MABN2d(64)
        """
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        
        s=compute_conv_output_size(32,5)
        s=s//2
        self.ksize.append(5)
        self.in_channel.append(3)
        self.map.append(s)
       
        s=compute_conv_output_size(s,5)
        s=s//2
        self.smid=s
        self.ksize.append(5)
        self.in_channel.append(64)
        self.map.append(64*self.smid*self.smid)
        self.map.extend([192])
        """
    def forward(self, x):
        if self.norm == 'batch_norm' or self.norm == 'mabn':
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
        elif self.norm == 'group_norm' or self.norm == 'instance_norm':
            x = self.pool(F.relu(self.gn1(self.conv1(x))))
            x = self.pool(F.relu(self.gn2(self.conv2(x))))
        else:
            #self.act['conv1'] = x
            x = self.pool(F.relu(self.conv1(x)))
            #self.act['conv2'] = x
            x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 5 * 5)

        #self.act['fc1'] = x
        x = F.relu(self.fc1(x))

        #self.act['fc2'] = x
        x = F.relu(self.fc2(x))

        #self.act['fc3'] = x
        x = self.fc3(x)

        return x
        

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.fc3 = nn.Linear(2048, 100, bias=False)
        
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        x = self.fc3(x)
        return x

import math

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


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()
        if net_act == 'sigmoid':
            self.net_act = nn.Sigmoid()
        elif net_act == 'relu':
            self.net_act = nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.net_act = None
            #exit('unknown activation function: %s'%net_act)

        if net_pooling == 'maxpooling'or net_pooling == 'cos_norm':
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling' :
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            self.net_pooling = None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        if net_pooling == 'cos_norm':
            self.classifier = ZeroMean_Classifier(num_feat, num_classes)
        else:
            self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            norm = nn.GroupNorm(2, shape_feat[0], affine=True)
        elif net_norm == 'none':
            norm = None
        else:
            norm = None
            exit('unknown net_norm: %s'%net_norm)
        return norm

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (24, 24)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            if self.net_act != None:
                layers += [self.net_act]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNet_Contrastive(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32),head='linear'):
        super(ConvNet_Contrastive, self).__init__()
        if net_act == 'sigmoid':
            self.net_act = nn.Sigmoid()
        elif net_act == 'relu':
            self.net_act = nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

        if net_pooling == 'maxpooling':
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            self.net_pooling = None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        feat_dim =128
        if head == 'cos_norm':
            self.classifier = CosNorm_Classifier(num_feat, num_classes)
        elif head == 'linear':
            self.head = nn.Linear(num_feat,feat_dim)
            self.classifier = nn.Linear(num_feat, num_classes)

        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(num_feat, num_feat),
                nn.ReLU(inplace=True),
                nn.Linear(num_feat, feat_dim)
            )
            self.classifier = nn.Linear(num_feat, num_classes)

        else:
            self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        feat = out.view(out.size(0), -1)
        norm_feat = F.normalize(self.head(feat), dim=1)

        out = self.classifier(feat)
        return out,norm_feat

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            norm = nn.GroupNorm(2, shape_feat[0], affine=True)
        elif net_norm == 'none':
            norm = None
        else:
            norm = None
            exit('unknown net_norm: %s'%net_norm)
        return norm

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (24, 24)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self.net_act]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat