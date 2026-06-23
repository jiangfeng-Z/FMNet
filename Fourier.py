#初始版本修改成0.5


import torch, time
import torch.nn as nn
import torch.nn.functional as F
from Rep import Rep

class Fourier_2D(nn.Module):
    def __init__(self, lamd) -> None:
        super(Fourier_2D, self).__init__()
        self.lamd = lamd

    def forward(self, input):
        b, c, n, h, w = input.shape
        dtype, device = input.dtype, input.device

        input =  torch.fft.fftshift(torch.fft.fftn(input.float(), dim=(3, 4), norm='ortho'))
        filter = self._CircleFilter(h, w, self.lamd).to(device)

        input = input * filter
        input = torch.fft.ifftn(torch.fft.ifftshift(input), s=(h, w), dim=(3, 4), norm='ortho').real.to(device)

        return input.type(dtype)

    def _CircleFilter(self, H, W, lamd):
        x_center, y_center = int(H // 2), int(W // 2)

        X, Y = torch.meshgrid(torch.arange(0, H, 1), torch.arange(0, W, 1), indexing='ij')
        circle = torch.sqrt((X - x_center)**2 + (Y - y_center)**2)
        r = min(H, W) // 32

        lp_F = (circle < r).to(torch.float32)
        hp_F = (circle > r).to(torch.float32)
        combined_Filter = lp_F * lamd + hp_F * (1 - lamd)
        combined_Filter[circle == r] = 1/3
        return combined_Filter
from thop import profile
from pytorch_memlab import MemReporter
# inputs = torch.randn((1, 3, 10, 512, 512)).cuda()
# focus_dists = torch.randn((1, 10, 512, 512)).cuda()
# model = Fourier_2D(0.8).cuda()
# flops, params = profile(model, inputs=(inputs,))
# print("flops: %.2fG, params: %.2fM"%(flops, params))#flops: 157302.13M, params: 1.89M

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, expand_ratio):
        # 参数验证
        if not isinstance(in_channel, int) or in_channel <= 0:
            raise ValueError("in_channel 必须是正整数")
        if not isinstance(out_channel, int) or out_channel <= 0:
            raise ValueError("out_channel 必须是正整数")
        if not isinstance(expand_ratio, int) or expand_ratio <= 0:
            raise ValueError("expand_ratio 必须是正整数")
        
        super(InvertedResidual, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.expand_ratio = expand_ratio
        
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = (in_channel == out_channel)
        
        self.conv = self.create_layers(hidden_channel)

    def create_layers(self, hidden_channel):
        layers = []
        if self.expand_ratio != 1:
            layers.append(self.create_expand_layer(self.in_channel, hidden_channel))
        
        layers.append(self.create_depthwise_layer(hidden_channel))
        layers.append(self.create_project_layer(hidden_channel, self.out_channel))
        
        return nn.Sequential(*layers)

    @staticmethod
    def create_expand_layer(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def create_depthwise_layer(channel):
        return nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=channel, bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def create_project_layer(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm3d(out_channel)
        )

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SepConv(nn.Module):
    def __init__(self, in_planes, group=1, use_bn=True, activation=nn.ReLU(inplace=True)) -> None:
        super(SepConv, self).__init__()
        self.spatial = nn.Conv3d(in_planes, in_planes, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=group, bias=not use_bn)
        self.temporal = nn.Conv3d(in_planes, in_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), groups=group, bias=not use_bn)
        
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
    def __init__(self, dim, lamda=0.5, use_bn=True, activation=nn.ReLU(inplace=True)):
        super(AFFBlock, self).__init__()
        self.dim = dim // 2
        self.lamda = lamda

        self.hf_filter = nn.Sequential(
            SepConv(self.dim, use_bn=use_bn, activation=activation),
            SepConv(self.dim, use_bn=use_bn, activation=activation),
        )
        self.lf_filter = Fourier_2D(self.lamda)

         # 融合
        self.fusion = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False), 
            nn.BatchNorm3d(dim) if use_bn else nn.Identity(),
            activation,
            nn.Conv3d(dim, dim,(3,1,1),stride=1,padding=(1,0,0),bias=False),
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