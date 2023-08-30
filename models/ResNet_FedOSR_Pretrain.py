# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 00:39:44 2022

@author: ZML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
model_urls = {
    'resnet18': './pre_model/resnet18-5c106cde.pth',
    'resnet34': './pre_model/resnet34-333f7ec4.pth',
    'resnet50': './pre_model/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MixLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super(MixLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x: torch.Tensor, targets = None, mode='softmax_threshold', fixed = False, alpha=0.8, is_detach = True, temperature = 1.,is_normalized = False):
        weight = self.weight

        outputs = F.linear(x, weight)

        if self.training and mode=='fedosr':
            
            prob = torch.nn.functional.softmax(outputs.detach()/temperature, dim=-1) # B C
            #targets # B
            fixed = fixed
            alpha = alpha
            if is_detach:
                weight_detach = weight.detach()
            else:
                weight_detach = weight
            mix_proto_batch = []
            for i in range(x.shape[0]):
                mix_proto_instance = []
                for c in range(weight.shape[0]):
                    if c != targets[i]:
                        if fixed == False:
                            alpha = prob[i][targets[i]]
                        mix_proto = alpha*weight_detach[targets[i]] + (1-alpha)*weight_detach[c]
                        mix_proto_instance.append(mix_proto) 
                mix_proto_instance = torch.stack(mix_proto_instance, dim = 0) # C-1, D
                mix_proto_batch.append(mix_proto_instance) #B, C-1, D
            mix_proto_batch = torch.stack(mix_proto_batch, dim = 0) #B, C-1, D
            if is_normalized:
                mix_out = torch.bmm(mix_proto_batch, x.unsqueeze(-1)).squeeze(-1) * 1.0/(weight.shape[0]-1) # B C-1
            else:
                mix_out = torch.bmm(mix_proto_batch, x.unsqueeze(-1)).squeeze(-1) # B C-1                
            outputs = torch.cat([outputs, mix_out], dim=1) # B, C + C-1

        return outputs

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.main_cls = nn.Linear(512, num_classes+1)
        
        self.auxiliary_layer4 = self._make_auxiliary_layer(block, 256, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])      
        self.auxiliary_cls = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_auxiliary_layer(self, block, inplanes, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        self.inplanes = inplanes    
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)    
    


    def aux_forward(self, x):
        x = self.auxiliary_layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        aux_out = self.auxiliary_cls(x)              
        out = {'aux_out':aux_out} 
        
        return out
    
    def discrete_forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)        
        outputs = self.main_cls(x)              
          
        out = {'outputs':outputs}
        
        return out  


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        aux_out = self.aux_forward(x.clone().detach())
        
        boundary_feats = x
        discrete_feats = x
        
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs = self.main_cls(x)
        
        out = {'outputs':outputs, 'aux_out':aux_out['aux_out'], 'boundary_feats': boundary_feats, 'discrete_feats':discrete_feats} 

        return out

def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = torch.load(model_urls[arch])
        net_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict)
        print(len(net_dict))
        print(len(pretrained_dict))   
        
    return model

def resnet18(pretrained=False, num_classes=8, **kwargs):

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, num_classes=num_classes,
                   **kwargs)


def resnet34(pretrained=False, num_classes=8, **kwargs):

    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, num_classes=num_classes,
                   **kwargs)


def resnet50(pretrained=False, num_classes=8, **kwargs):

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, num_classes=num_classes,
                   **kwargs)


if __name__=='__main__':
    net = resnet18(False, 3)
    x = torch.randn(2,3,144,144)
    targets = torch.Tensor([1,0]).long()
    y = net(x)
    #print(y)
