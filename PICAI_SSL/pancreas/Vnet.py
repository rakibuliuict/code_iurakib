# import torch
# from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt
# import pdb


# class ConvBlock(nn.Module):
#     def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
#         super(ConvBlock, self).__init__()

#         ops = []
#         for i in range(n_stages):
#             if i == 0:
#                 input_channel = n_filters_in
#             else:
#                 input_channel = n_filters_out

#             ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
#             if normalization == 'batchnorm':
#                 ops.append(nn.BatchNorm3d(n_filters_out))
#             elif normalization == 'groupnorm':
#                 ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#             elif normalization == 'instancenorm':
#                 ops.append(nn.InstanceNorm3d(n_filters_out))
#             elif normalization != 'none':
#                 assert False
#             ops.append(nn.ReLU(inplace=True))

#         self.conv = nn.Sequential(*ops)

#     def forward(self, x):
#         x = self.conv(x)
#         return x
    
    
# class DownsamplingConvBlock(nn.Module):
#     def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
#         super(DownsamplingConvBlock, self).__init__()

#         ops = []
#         if normalization != 'none':
#             ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
#             if normalization == 'batchnorm':
#                 ops.append(nn.BatchNorm3d(n_filters_out))
#             elif normalization == 'groupnorm':
#                 ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#             elif normalization == 'instancenorm':
#                 ops.append(nn.InstanceNorm3d(n_filters_out))
#             else:
#                 assert False
#         else:
#             ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

#         ops.append(nn.ReLU(inplace=True))

#         self.conv = nn.Sequential(*ops)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class UpsamplingDeconvBlock(nn.Module):
#     def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
#         super(UpsamplingDeconvBlock, self).__init__()

#         ops = []
#         if normalization != 'none':
#             ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
#             if normalization == 'batchnorm':
#                 ops.append(nn.BatchNorm3d(n_filters_out))
#             elif normalization == 'groupnorm':
#                 ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#             elif normalization == 'instancenorm':
#                 ops.append(nn.InstanceNorm3d(n_filters_out))
#             else:
#                 assert False
#         else:

#             ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

#         ops.append(nn.ReLU(inplace=True))

#         self.conv = nn.Sequential(*ops)

#     def forward(self, x):
#         x = self.conv(x)
#         return x
    
    
# class VNet(nn.Module):
#     def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='instancenorm', has_dropout=False):
#         super(VNet, self).__init__()
#         self.has_dropout = has_dropout

#         self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
#         self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

#         self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

#         self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

#         self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

#         self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
#         self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

#         self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

#         self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

#         self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
#         if has_dropout:
#             self.dropout = nn.Dropout3d(p=0.5)
#         self.branchs = nn.ModuleList()
#         for i in range(1):
#             if has_dropout:
#                 seq = nn.Sequential(
#                     ConvBlock(1, n_filters, n_filters, normalization=normalization),
#                     nn.Dropout3d(p=0.5),
#                     nn.Conv3d(n_filters, n_classes, 1, padding=0)
#                 )
#             else:
#                 seq = nn.Sequential(
#                     ConvBlock(1, n_filters, n_filters, normalization=normalization),
#                     nn.Conv3d(n_filters, n_classes, 1, padding=0)
#                 )
#             self.branchs.append(seq)

#     def encoder(self, input):
#         x1 = self.block_one(input)
#         x1_dw = self.block_one_dw(x1)

#         x2 = self.block_two(x1_dw)
#         x2_dw = self.block_two_dw(x2)

#         x3 = self.block_three(x2_dw)
#         x3_dw = self.block_three_dw(x3)

#         x4 = self.block_four(x3_dw)
#         x4_dw = self.block_four_dw(x4)

#         x5 = self.block_five(x4_dw)

#         if self.has_dropout:
#             x5 = self.dropout(x5)

#         res = [x1, x2, x3, x4, x5]

#         return res

#     def decoder(self, features):
#         x1 = features[0]
#         x2 = features[1]
#         x3 = features[2]
#         x4 = features[3]
#         x5 = features[4]

#         x5_up = self.block_five_up(x5)
#         x5_up = x5_up + x4

#         x6 = self.block_six(x5_up)
#         x6_up = self.block_six_up(x6)
#         x6_up = x6_up + x3

#         x7 = self.block_seven(x6_up)
#         x7_up = self.block_seven_up(x7)
#         x7_up = x7_up + x2

#         x8 = self.block_eight(x7_up)
#         x8_up = self.block_eight_up(x8)
#         x8_up = x8_up + x1
#         out = []
#         for branch in self.branchs:
#             o = branch(x8_up)
#             out.append(o)
#         out.append(x6)
#         return out

#     def forward(self, input, turnoff_drop=False):
#         if turnoff_drop:
#             has_dropout = self.has_dropout
#             self.has_dropout = False
#         features = self.encoder(input)
#         out = self.decoder(features)
#         if turnoff_drop:
#             self.has_dropout = has_dropout
#         return out

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args
from monai.utils import deprecated_arg

__all__ = ["VNet"]

def get_acti_layer(act: tuple[str, dict] | str, nchan: int = 0):
    if act == "prelu":
        act = ("prelu", {"num_parameters": nchan})
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)

class LUConv(nn.Module):
    def __init__(self, spatial_dims: int, nchan: int, act: tuple[str, dict] | str, bias: bool = False):
        super().__init__()
        self.act_function = get_acti_layer(act, nchan)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.act_function(out)
        return out

def _make_nconv(spatial_dims: int, nchan: int, depth: int, act: tuple[str, dict] | str, bias: bool = False):
    return nn.Sequential(*[LUConv(spatial_dims, nchan, act, bias) for _ in range(depth)])

class InputTransition(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.act_function = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )
        self.adapter = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv_block(x)
        x_proj = self.adapter(x)
        out = self.act_function(torch.add(out, x_proj))
        return out

class DownTransition(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, nconvs: int, act: tuple[str, dict] | str, dropout_prob: float | None = None, dropout_dim: int = 3, bias: bool = False):
        super().__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]
        dropout_type = Dropout[Dropout.DROPOUT, dropout_dim]

        out_channels = 2 * in_channels
        self.down_conv = conv_type(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.bn1 = norm_type(out_channels)
        self.act_function1 = get_acti_layer(act, out_channels)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act, bias)
        self.dropout = dropout_type(dropout_prob) if dropout_prob is not None else None

    def forward(self, x):
        down = self.act_function1(self.bn1(self.down_conv(x)))
        out = self.dropout(down) if self.dropout else down
        out = self.ops(out)
        out = self.act_function2(torch.add(out, down))
        return out

class UpTransition(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, nconvs: int, act: tuple[str, dict] | str, dropout_prob: tuple[float | None, float] = (None, 0.5), dropout_dim: int = 3):
        super().__init__()
        conv_trans_type = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]
        dropout_type = Dropout[Dropout.DROPOUT, dropout_dim]

        self.up_conv = conv_trans_type(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn1 = norm_type(out_channels // 2)
        self.dropout = dropout_type(dropout_prob[0]) if dropout_prob[0] is not None else None
        self.dropout2 = dropout_type(dropout_prob[1])
        self.act_function1 = get_acti_layer(act, out_channels // 2)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act)

    def forward(self, x, skipx):
        x = self.dropout(x) if self.dropout else x
        skipx = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(x)))
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.act_function2(torch.add(out, xcat))
        return out

class OutputTransition(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False):
        super().__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        self.act_function1 = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )
        self.conv2 = conv_type(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.act_function1(out)
        out = self.conv2(out)
        return out

class VNet(nn.Module):
    @deprecated_arg(name="dropout_prob", since="1.2", new_name="dropout_prob_down")
    @deprecated_arg(name="dropout_prob", since="1.2", new_name="dropout_prob_up")
    def __init__(self, spatial_dims: int = 3, in_channels: int = 1, out_channels: int = 1, act: tuple[str, dict] | str = ("elu", {"inplace": True}), dropout_prob: float | None = 0.5, dropout_prob_down: float | None = 0.5, dropout_prob_up: tuple[float | None, float] = (0.5, 0.5), dropout_dim: int = 3, bias: bool = False):
        super().__init__()
        self.in_tr = InputTransition(spatial_dims, in_channels, 16, act, bias=bias)
        self.down_tr32 = DownTransition(spatial_dims, 16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(spatial_dims, 32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(spatial_dims, 64, 3, act, dropout_prob=dropout_prob_down, bias=bias)
        self.down_tr256 = DownTransition(spatial_dims, 128, 2, act, dropout_prob=dropout_prob_down, bias=bias)
        self.up_tr256 = UpTransition(spatial_dims, 256, 256, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr128 = UpTransition(spatial_dims, 256, 128, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr64 = UpTransition(spatial_dims, 128, 64, 1, act)
        self.up_tr32 = UpTransition(spatial_dims, 64, 32, 1, act)
        self.out_tr = OutputTransition(spatial_dims, 32, out_channels, act, bias=bias)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        return self.out_tr(x)
