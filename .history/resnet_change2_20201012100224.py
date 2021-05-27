from __future__ import absolute_import
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

from prune_params import ResNet20_Channel_Prune
prune_index = ResNet20_Channel_Prune.index
prune_ratio = ResNet20_Channel_Prune.prune_ratio

channel16 = list(range(2, 17, 2))
channel32 = list(range(2, 33, 2))
channel64 = list(range(2, 65, 2))

#__all__ = ['resnet']

def get_remained_filters(weight_torch, prune_ratio):
    length = np.prod(weight_torch.size())
    mask = torch.zeros(weight_torch.shape[0])
    filter_remained_num = 0
    if len(weight_torch.size()) == 4:
        filter_remained_num = max(int(round((weight_torch.shape[0] * (1 - prune_ratio)) / 2) * 2), 2)
        weight_vec = weight_torch.view(weight_torch.shape[0], -1)
        norm2 = torch.norm(weight_vec, 2, 1)
        norm2_np = norm2.cpu().numpy()
        filter_index = norm2_np.argsort()[::-1][:filter_remained_num]
        assert filter_index.size == filter_remained_num, 'size of remainde filter num not correct'
        mask[filter_index.tolist()] = 1
    else:
        pass
    return mask, filter_remained_num

def channel_shuffle(x, groups=4):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, index_p, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.cfg = cfg
        self.planes = planes
        self.index_p = index_p

    def forward(self, x, weights):
        #global weights
        residual = x

        choose_channels = []
        if self.planes == 16:
            choose_channels = list(range(2, 17, 2))
        elif self.planes == 32:
            choose_channels = list(range(2, 33, 2))
        else:
            choose_channels = list(range(2, 65, 2))

        out_channels = self.cfg
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        choices = [choose_channels[prune_ratio[self.index_p][j]] for j in range(4)]
        #choices = [max(int(round((out_channels * (1 - prune_ratio[self.index_p][j])) / 2) * 2), 2) for j in range(5)]
        out_convs = [out[:, :oC] for oC in choices]
        outA = ChannelWiseInterV2(out_convs[0], out_channels)
        outB = ChannelWiseInterV2(out_convs[1], out_channels)
        outC = ChannelWiseInterV2(out_convs[2], out_channels)
        outD = ChannelWiseInterV2(out_convs[3], out_channels)
        #outE = ChannelWiseInterV2(out_convs[4], out_channels)
        out = outA * weights[self.index_p][0] + outB * weights[self.index_p][1] + outC * weights[self.index_p][2] + outD * weights[self.index_p][3]

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class ResNet(nn.Module):

    def __init__(self, depth, dataset='cifar10', cfg=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[16], [16]*n, [32]*n, [64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg
        self.n = n
        self.depth = depth

        self._initialize_alphas()
        self.inplanes = cfg[0]
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, 0, cfg=cfg[1:n+1])
        self.layer2 = self._make_layer(block, 32, n, 1, cfg=cfg[n+1:2*n+1], stride=2)
        self.layer3 = self._make_layer(block, 64, n, 2, cfg=cfg[2*n+1:3*n+1], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, index, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, index, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, index, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        #global weights
        weights = F.softmax(self.arch_params, dim=-1)
        weights = weights.cuda(torch.cuda.current_device())

        out = self.conv1(x)
        out = self.bn1(out)
        #x = self.bn1(out)
        x = self.relu(out)    # 32x32

        for j in range(self.n):
            x = self.layer1[j](x, weights)
        for j in range(self.n):
            x = self.layer2[j](x, weights)
        for j in range(self.n):
            x = self.layer3[j](x, weights)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _initialize_alphas(self):
        k = len(prune_index)
        self.arch_params = Variable(1e-3 * torch.randn(k, 4).cuda(), requires_grad=True)

    def arch_parameters(self):
        return [self.arch_params]

    def new(self):
        model_new = ResNet(self.depth).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

def print_log(print_str, log):
    log.write('{}\n'.format(print_str))
    log.flush()

def ChannelWiseInterV2(inputs, oC):
    assert inputs.dim() == 4, 'invalid dimension : {:}'.format(inputs.size())
    batch, C, H, W = inputs.size()
    inputs_5D = inputs.view(batch, 1, C, H, W)
    outputs_5D = nn.functional.interpolate(inputs_5D, (oC,H,W), None, 'area', None)
    #otputs    = otputs_5D.view(batch, oC, H, W)
    #outputs_5D = nn.functional.interpolate(inputs_5D, (oC,H,W), None, 'trilinear', False)
    outputs = outputs_5D.view(batch, oC, H, W)
    return outputs

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

if __name__ == '__main__':
    net = resnet(depth=32)
    for k, m in enumerate(net.modules()):
        print(k)
        print(m)
    x=Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)
