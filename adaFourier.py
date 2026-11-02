# -*- coding: utf-8 -*-
"""FMNet with three adaptive FMTM locations (2026-11-02).

Usage:
    from adaFourier import Network, FasterNet_model_convert
    model = Network()
    coarse, middle, final = model(focal_stack, focus_positions)

The input is [B, 3, N, H, W]. Pad H/W to multiples of 32 as in the
original data loaders. All three predictions are [B, H, W]. The initial
H x W stage is only Conv3d + BatchNorm3d + ReLU: no FMTM and no FFT.

Paper S1: feature_extraction2, H/2 x W/2 (8 Fourier-branch channels).
Paper S2: feature_extraction3, H/4 x W/4 (16 Fourier-branch channels).
Paper MSP: fuseModule.aff, H/4 x W/4 (16 Fourier-branch channels).
The three modules have independent scalar alpha/beta pairs (6 scalars).
The optional last_lamd values are per-module statistics from the most
recent forward batch, not an evaluation-set average or a stored optimum.

The original FED, MSP convolution paths, decoder and three-output order
are retained. Definitions of Rep, SepConv and the backbone are copied
from the project. This module is self-contained: it does not import the
old Fourier.py, Fourierada.py, module.py or Rep.py.

The low/high masks partition the spatial frequency plane exactly as
paper Eq. (21): distance <= D and > D, D=min(H,W)/32. There is no legacy
1/3-weight boundary ring. The existing input-conditioned lambda formula,
detached energy ratio and default lambda interval [0.5,0.95] are retained.

This changes the four-FMTM architecture. Old four-FMTM checkpoints are
not directly compatible and do not provide measured three-FMTM results.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv3d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding, 
                                        groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm3d(num_features=out_channels))
    return result


class Attention(nn.Module):
    def __init__(self, dim) -> None:
        super(Attention, self).__init__()
        self.combine = nn.Conv3d(dim, dim, (1, 2, 1), stride=1, padding=0, groups=dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, 4*dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim*4, dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, n, h, w = x.shape
        y = torch.fft.fftn(x, dim=(3, 4), norm='ortho').real
        avg_y = F.adaptive_avg_pool3d(y, output_size=(None, 1, 1))#(b, c, n, 1, 1)
        max_y = F.adaptive_max_pool3d(y, output_size=(None, 1, 1))#(b, c, n, 1, 1)
        combine_y = torch.cat([avg_y, max_y], dim=-2)#(b, c, n, 2, 1)
        result_y = self.combine(combine_y)#(b, c, n, 1, 1)
        channels_weight = self.mlp(result_y)
        out = x * channels_weight.expand_as(x)
        return out


class Rep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, use_nonlinearity=True,
                 padding_mode='zeros', deploy=False, use_se=False):
        super(Rep, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.kernerl = kernel_size

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        if use_nonlinearity:
            self.nonlinearity = nn.ReLU(inplace=True)
        else:
            self.nonlinearity = nn.Identity()
    
        if use_se:
            self.se = Attention(out_channels)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, 
                                         bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm3d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=kernel_size, stride=stride, 
                                     padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, 
                                   kernel_size=1, stride=stride, 
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        se = self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        return self.nonlinearity(se)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None: 
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm3d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv3d(in_channels=self.rbr_dense.conv.in_channels, 
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, 
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, 
                                     dilation=self.rbr_dense.conv.dilation, 
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class Fourier_2D(nn.Module):
    """Input-conditioned channel coefficients; alpha/beta shared within one module."""

    def __init__(self, lamd=0.8, lamd_min=0.5, lamd_max=0.95, D0=32, eps=1e-6):
        super().__init__()
        if not (0 <= lamd_min < lamd_max <= 1):
            raise ValueError('Expected 0 <= lamd_min < lamd_max <= 1')
        if D0 <= 0:
            raise ValueError('D0 must be positive')
        self.lamd_min, self.lamd_max = float(lamd_min), float(lamd_max)
        self.D0, self.eps = D0, eps
        lamd = max(min(lamd, lamd_max - 1e-4), lamd_min + 1e-4)
        initial = (lamd - lamd_min) / (lamd_max - lamd_min)
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(math.log(initial / (1 - initial)), dtype=torch.float32))
        self._last_lamd = None
        self.last_lamd_per_channel = None

    @property
    def last_lamd(self):
        """Mean over this module's most recent batch and Fourier-branch channels."""
        return self._last_lamd

    def _CircleFilter(self, H, W, device, dtype):
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        distance = torch.sqrt((y.float() - H // 2) ** 2 + (x.float() - W // 2) ** 2)
        radius = min(H, W) / self.D0
        low = (distance <= radius).to(dtype).view(1, 1, 1, H, W)
        high = (distance > radius).to(dtype).view(1, 1, 1, H, W)
        # Keep the historical three-return helper interface; no third band exists.
        return low, high, torch.zeros_like(low)

    def forward(self, input):
        if input.ndim != 5:
            raise ValueError('Fourier_2D expects [B,C,N,H,W]')
        _, _, frames, height, width = input.shape
        freq = torch.fft.fftshift(torch.fft.fftn(input.float(), dim=(3, 4), norm='ortho'), dim=(3, 4))
        low, high, _ = self._CircleFilter(height, width, input.device, freq.real.dtype)
        power = freq.abs().pow(2)
        low_energy = (power * low).sum(dim=(2, 3, 4), keepdim=True) / (low.sum() * frames + self.eps)
        high_energy = (power * high).sum(dim=(2, 3, 4), keepdim=True) / (high.sum() * frames + self.eps)
        ratio = (low_energy / (low_energy + high_energy + self.eps)).detach()
        lamd = self.lamd_min + (self.lamd_max - self.lamd_min) * torch.sigmoid(
            self.alpha * (ratio - 0.5) + self.beta)
        self.last_lamd_per_channel = lamd.detach().reshape(input.shape[:2]).clone()
        self._last_lamd = lamd.detach().mean().item()
        filtered = freq * (low * lamd + high * (1 - lamd))
        result = torch.fft.ifftn(torch.fft.ifftshift(filtered, dim=(3, 4)),
                                 s=(height, width), dim=(3, 4), norm='ortho').real
        return result.to(dtype=input.dtype)


class SepConv(nn.Module):
    def __init__(self, in_planes, group=1, use_bn=True, activation=nn.ReLU(inplace=True)) -> None:
        super(SepConv, self).__init__()

        self.spatial = nn.Conv3d(
            in_planes,
            in_planes,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            groups=group,
            bias=not use_bn
        )

        self.temporal = nn.Conv3d(
            in_planes,
            in_planes,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            groups=group,
            bias=not use_bn
        )

        if use_bn:
            self.bn = nn.BatchNorm3d(in_planes)
        else:
            self.bn = nn.Identity()

        self.activation = activation

    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class AFFBlock(nn.Module):
    def __init__(self, dim, lamda=0.8, use_bn=True, activation=nn.ReLU(inplace=True)):
        super(AFFBlock, self).__init__()

        self.dim = dim // 2
        self.lamda = lamda

        self.hf_filter = nn.Sequential(
            SepConv(self.dim, use_bn=use_bn, activation=activation),
            SepConv(self.dim, use_bn=use_bn, activation=activation),
        )

        self.lf_filter = Fourier_2D(lamd=self.lamda)

        self.fusion = nn.Sequential(
            nn.Conv3d(
                dim,
                dim,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(dim) if use_bn else nn.Identity(),
            activation,
            nn.Conv3d(
                dim,
                dim,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
                bias=False
            ),
            nn.BatchNorm3d(dim) if use_bn else nn.Identity(),
            activation
        )

    def forward(self, x):
        hf, lf = torch.chunk(x, 2, dim=1)

        hf_out = self.hf_filter(hf)
        lf_out = self.lf_filter(lf)

        out = torch.cat([hf_out, lf_out], dim=1)
        out = self.fusion(out + x)

        return out


class FuseLayer(nn.Module):
    def __init__(self, in_planes) -> None:
        super(FuseLayer, self).__init__()
        self.pool_8 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool_16 = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.pool_32 = nn.MaxPool3d(kernel_size=(1, 8, 8), stride=(1, 8, 8))

        self.aff = AFFBlock(in_planes)
        self.conv8_0 = nn.Sequential(
            Rep(in_planes, in_planes, 3, 1, 1),
            Rep(in_planes, in_planes, 3, 1, 1)
        )
        self.conv8_1 = nn.Sequential(
            Rep(in_planes, in_planes, 3, 1, 1),
            Rep(in_planes, in_planes, 3, 1, 1, use_nonlinearity=False)
        )

        self.conv16_0 = nn.Sequential(
            Rep(in_planes, 2*in_planes, 3, 1, 1),
            Rep(2*in_planes, 2*in_planes, 3, 1, 1)
        )
        self.conv16_1 = nn.Sequential(
            Rep(2*in_planes, 2*in_planes, 3, 1, 1),
            Rep(2*in_planes, 2*in_planes, 3, 1, 1, use_nonlinearity=False)
        )

        self.conv32_0 = nn.Sequential(
            Rep(in_planes, 2*in_planes, 3, 1, 1),
            Rep(2*in_planes, 2*in_planes, 3, 1, 1)
        )
        self.conv32_1 = nn.Sequential(
            Rep(2*in_planes, 2*in_planes, 3, 1, 1),
            Rep(2*in_planes, 2*in_planes, 3, 1, 1, use_nonlinearity=False)
        )

        self.conv_1_down = nn.Conv3d(in_planes, in_planes, 3, (1, 2, 2), 1, bias=False)
        self.conv_8_down = nn.Conv3d(in_planes, 2*in_planes, 3, (1, 2, 2), 1, bias=False)
        self.conv_16_down = nn.Conv3d(in_planes*2, in_planes*2, 3, (1, 2, 2), 1, bias=False)

        self.combine1 = nn.Sequential(
            Rep(in_planes, in_planes, 3, 1, 1),
            Rep(in_planes, in_planes, 3, 1, 1)
        )
        self.combine2 = nn.Sequential(
            Rep(in_planes*2, in_planes*2, 3, 1, 1),
            Rep(in_planes*2, in_planes*2, 3, 1, 1)
        )
        self.combine3 = nn.Sequential(
            Rep(in_planes*2, in_planes*2, 3, 1, 1),
            Rep(in_planes*2, in_planes*2, 3, 1, 1)
        )

        self.out = nn.Sequential(
            Rep(in_planes*6, in_planes, 3, 1, 1),
            Rep(in_planes, in_planes, 3, 1, 1)
        )
    
    def forward(self, input):#32
        b, c, n, h, w = input.shape
        out_8 = self.pool_8(input)
        out_16 = self.pool_16(input)
        out_32 = self.pool_32(input)

        out_1 = self.aff(input)

        out_8_res = self.conv8_0(out_8)
        out_8 = F.relu(self.conv8_1(out_8_res) + out_8_res)

        out_16_res = self.conv16_0(out_16)
        out_16 = F.relu(self.conv16_1(out_16_res) + out_16_res)

        out_32_res = self.conv32_0(out_32)
        out_32 = F.relu(self.conv32_1(out_32_res) + out_32_res)
        
        out_1_1 = self.conv_1_down(out_1)
        out_8_1 = out_1_1 + out_8
        out_8_1 = self.combine1(out_8_1)

        out_16_1 = self.conv_8_down(out_8_1)
        out_16_1 = out_16_1 + out_16
        out_16_1 = self.combine2(out_16_1)

        out_32_1 = self.conv_16_down(out_16_1)
        out_32_1 = out_32_1 + out_32
        out_32_1 = self.combine3(out_32_1)

        out_32_8 = F.interpolate(out_32_1, size=[n, h, w], mode='trilinear')
        out_16_8 = F.interpolate(out_16_1, size=[n, h, w], mode='trilinear')
        out_8_8 = F.interpolate(out_8_1, size=[n, h, w], mode='trilinear')

        final = torch.cat([out_1, out_8_8, out_16_8, out_32_8], dim=1)
        out = self.out(final)
        return out


class Down(nn.Module):
    def __init__(self, in_planes, out_planes, downsample_factor=(1, 2, 2), use_bn=True, activation=nn.ReLU(inplace=True)) -> None:
        super(Down, self).__init__()
        self.downsample_factor = downsample_factor
        self.max_pool = nn.MaxPool3d(kernel_size=downsample_factor, stride=downsample_factor)
        self.conv_pool = nn.Conv3d(in_planes, in_planes, kernel_size=(3, 3, 3), stride=downsample_factor, padding=(1, 1, 1))
        self.combine = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm3d(out_planes)
        self.activation = activation

    def forward(self, x):
        max_pooled = self.max_pool(x)
        conv_pooled = self.conv_pool(x)
        combined = max_pooled + conv_pooled
        out = self.combine(combined)
        if self.use_bn:
            out = self.bn(out)
        out = self.activation(out)
        return out


class Out(nn.Module):
    def __init__(self, in_planes, activation=nn.Softplus(beta=1, threshold=20)) -> None:
        super(Out, self).__init__()
        self.out = nn.Conv3d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation = activation

    def forward(self, input, focus_dists, Height, Width):
        B, C, N, H, W = input.shape
        input = self.out(input)
        input = torch.squeeze(input, 1)

        if Height is not None and Width is not None and (H != Height or W != Width):
            input = F.interpolate(input, size=(Height, Width), mode='bilinear', align_corners=True)

        input = self.activation(input) + 1e-6
        input = input / input.sum(axis=1, keepdim=True)
        out = torch.sum(focus_dists * input, dim=1)

        return out


class Feature_Extraction(nn.Module):
    def __init__(self, in_planes) -> None:
        super(Feature_Extraction, self).__init__()
        self.block = AFFBlock(in_planes)

    def forward(self, input):
        return self.block(input)


class AggregationModule(nn.Module):
    def __init__(self, in_planes) -> None:
        super(AggregationModule, self).__init__()
        self.dim = in_planes // 2
        self.compress = nn.Sequential(
            nn.Conv3d(in_planes, self.dim, 1, 1, bias=False),
            nn.BatchNorm3d(self.dim),
            nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Sequential(
            Rep(self.dim, self.dim, 3, 1, 1),
            Rep(self.dim, self.dim, 3, 1, 1),
            Rep(self.dim, self.dim, 3, 1, 1),
        )

    def forward(self, input):
        out1 = self.compress(input)
        out = self.conv(out1)
        return out


class Network(nn.Module):
    def __init__(self, in_planes = 3, base = 8) -> None:
        super(Network, self).__init__()
        self.feature_extraction1 = nn.Sequential(
            nn.Conv3d(in_planes, base, kernel_size=(1, 9, 9), stride=1, padding=(0, 8, 8), dilation=(1, 2, 2)),
            nn.BatchNorm3d(base),
            nn.ReLU(inplace=True),
        )#3->8

        self.feature_extraction2 = nn.Sequential( 
            Down(in_planes=base, out_planes=base*2),
            Feature_Extraction(base*2)
            )#8->16
        
        self.feature_extraction3 = nn.Sequential(
            Down(in_planes=base*2, out_planes=base*4),
            Feature_Extraction(base*4)#32->32
            )#16-32
        
        self.fuseModule = FuseLayer(32)
        self.out_mid = Out(32)
        
        self.AggregationModule2 = AggregationModule(48)
        self.out_2 = Out(24)

        self.AggregationModule3 = AggregationModule(32)
        self.out_3 = Out(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, input, focus):
        b, c, n, h, w = input.shape
        feature_1 = self.feature_extraction1(input)
        feature_2 = self.feature_extraction2(feature_1)
        feature_4 = self.feature_extraction3(feature_2)

        feature_4_r = self.fuseModule(feature_4)
        out_4 = self.out_mid(feature_4_r, focus, h, w)

        feature_2_r = F.interpolate(feature_4_r, size=[n, feature_2.shape[3], feature_2.shape[4]], mode='trilinear')
        feature_2_r = torch.cat([feature_2, feature_2_r], dim=1)
        feature_2_r = self.AggregationModule2(feature_2_r)
        out_2 = self.out_2(feature_2_r, focus, h, w)

        feature_1_r = F.interpolate(feature_2_r, size=[n, feature_1.shape[3], feature_1.shape[4]], mode='trilinear')
        feature_1_r = torch.cat([feature_1, feature_1_r], dim=1)
        feature_1_r = self.AggregationModule3(feature_1_r)
        out_1 = self.out_3(feature_1_r, focus, h, w)
        return out_4, out_2, out_1


def FasterNet_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model, save_path)
    return model


FMNet = Network
FMTM = AFFBlock


def get_adaptive_lamda_info(model):
    """Return the unweighted mean of the three last-batch stage means and details.

    This is not the mean over an entire dataset. Accumulate per input in the
    evaluation loop if a dataset average is required.
    """
    net = model.module if isinstance(model, nn.DataParallel) else model
    values, details = [], []
    for name, module in net.named_modules():
        if isinstance(module, Fourier_2D) and module.last_lamd is not None:
            values.append(module.last_lamd)
            details.append('{}:{:.6f}'.format(name, module.last_lamd))
    return (sum(values) / len(values), '; '.join(details)) if values else (None, 'None')
