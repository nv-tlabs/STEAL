# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# ResNet implementation kindly borrow from pytorch-vision and modified to match the original casenet caffe implementation.

import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np

__all__ = ['casenet101']

BatchNorm = nn.BatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_no, stride=1, downsample=None):  # add dilation factor
        super(Bottleneck, self).__init__()
        if block_no < 5:
            dilation = 2
            padding = 2
        else:
            dilation = 4
            padding = 4

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices).long()
            x = x.index_select(axis, Variable(indices))
        return x


class MyIdentity(nn.Module):
    def __init__(self, axis, offset):
        super(MyIdentity, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        return x


class SideOutputCrop(nn.Module):
    """
    This is the original implementation ConvTranspose2d (fixed) and crops
    """

    def __init__(self, num_output, kernel_sz=None, stride=None, upconv_pad=0, do_crops=True):
        super(SideOutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(num_output, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(1, out_channels=1, kernel_size=kernel_sz, stride=stride,
                                                padding=upconv_pad,
                                                bias=False)
            ##doing crops
            if self._do_crops:
                self.crops = Crop(2, offset=kernel_sz // 4)
            else:
                self.crops = MyIdentity(None, None)
        else:
            self.upsample = False

    def forward(self, res, reference=None):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)
            side_output = self.crops(side_output, reference)

        return side_output


class Res5OutputCrop(nn.Module):

    def __init__(self, in_channels=2048, kernel_sz=16, stride=8, nclasses=20, upconv_pad=0, do_crops=True):
        super(Res5OutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(in_channels, nclasses, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsampled = nn.ConvTranspose2d(nclasses, out_channels=nclasses, kernel_size=kernel_sz, stride=stride,
                                            padding=upconv_pad,
                                            bias=False, groups=nclasses)
        if self._do_crops is True:
            self.crops = Crop(2, offset=kernel_sz // 4)
        else:
            self.crops = MyIdentity(None, None)

    def forward(self, res, reference):
        res = self.conv(res)
        res = self.upsampled(res)
        res = self.crops(res, reference)
        return res


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class ResNet(nn.Module):

    def __init__(self, block, layers, nclasses=20):

        self.inplanes = 64
        super(ResNet, self).__init__()
        self._nclasses = nclasses
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)

        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # define ceil mode

        self.layer1 = self._make_layer(block, 64, layers[0], 2)  # res2
        self.layer2 = self._make_layer(block, 128, layers[1], 3, stride=2)  # res3

        self.layer3 = self._make_layer(block, 256, layers[2], 4, stride=2)  # res4
        self.layer4 = self._make_layer(block, 512, layers[3], 5, stride=1)  # res5

        self.normals = None

        ####let's make pointers to keep compatibility for now

        SideOutput_fn = SideOutputCrop
        Res5Output_fn = Res5OutputCrop

        self.use_pytorch_upsample = False

        # The original casenet implementation has padding when upsampling cityscapes that are not used for SBD.
        # Leaving like this for now such that it is clear and it matches the original implementation (for fair comparison)

        if nclasses == 19:
            print('Assuming Cityscapes CASENET')
            self.score_edge_side1 = SideOutput_fn(64)
            self.score_edge_side2 = SideOutput_fn(256, kernel_sz=4, stride=2, upconv_pad=1, do_crops=False)
            self.score_edge_side3 = SideOutput_fn(512, kernel_sz=8, stride=4, upconv_pad=2, do_crops=False)
            self.score_cls_side5 = Res5Output_fn(kernel_sz=16, stride=8, nclasses=self._nclasses, upconv_pad=4,
                                                 do_crops=False)
        else:
            print('Assuming Classical SBD CASENET')
            self.score_edge_side1 = SideOutput_fn(64)
            self.score_edge_side2 = SideOutput_fn(256, kernel_sz=4, stride=2)
            self.score_edge_side3 = SideOutput_fn(512, kernel_sz=8, stride=4)
            self.score_cls_side5 = Res5Output_fn(kernel_sz=16, stride=8, nclasses=self._nclasses)

        num_classes = self._nclasses
        self.ce_fusion = nn.Conv2d(4 * num_classes, num_classes, groups=num_classes, kernel_size=1, stride=1, padding=0,
                                   bias=True)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # manually initializing the new layers.
        self.score_edge_side1.conv.weight.data.normal_(0, 0.01)
        self.score_edge_side1.conv.bias.data.zero_()
        # -
        self.score_edge_side2.conv.weight.data.normal_(0, 0.01)
        self.score_edge_side2.conv.bias.data.zero_()
        # -
        self.score_edge_side3.conv.weight.data.normal_(0, 0.01)
        self.score_edge_side3.conv.bias.data.zero_()
        # -
        self.ce_fusion.weight.data.fill_(0.25)
        self.ce_fusion.bias.data.zero_()
        # ---

    def _make_layer(self, block, planes, blocks, block_no, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, block_no, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_no))

        return nn.Sequential(*layers)

    def _sliced_concat(self, res1, res2, res3, res5, num_classes):
        out_dim = num_classes * 4
        out_tensor = Variable(torch.FloatTensor(res1.size(0), out_dim, res1.size(2), res1.size(3))).cuda()
        class_num = 0
        for i in range(0, out_dim, 4):
            out_tensor[:, i, :, :] = res5[:, class_num, :, :]
            out_tensor[:, i + 1, :, :] = res1[:, 0, :, :]  # it needs this trick for multibatch
            out_tensor[:, i + 2, :, :] = res2[:, 0, :, :]
            out_tensor[:, i + 3, :, :] = res3[:, 0, :, :]

            class_num += 1

        return out_tensor

    def forward(self, x, normals_mask=None):
        assert x.shape[1] == 3, 'N,3,H,W BGR Image?'
        input_data = x

        # res1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        side_1 = self.score_edge_side1(x, input_data)

        # res2
        x = self.maxpool(x)
        x = self.layer1(x)
        side_2 = self.score_edge_side2(x, input_data)

        # res3
        x = self.layer2(x)
        side_3 = self.score_edge_side3(x, input_data)

        # res4
        x = self.layer3(x)

        # res5
        x = self.layer4(x)

        side_5 = self.score_cls_side5(x, input_data)

        # combine outputs and classify
        sliced_cat = self._sliced_concat(side_1, side_2, side_3, side_5, self._nclasses)
        acts = self.ce_fusion(sliced_cat)

        normals = None

        return [acts, side_5, normals, (side_1, side_2, side_3)]  # sigmoid can be taken later


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def casenet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
