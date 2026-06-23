# -*- coding: utf-8 -*-
#自适应lamda，训练过程中自适应调整，限制范围0.5-0.95，默认0.7，训练过程中自适应调整
import torch, time, math
import torch.nn as nn
import torch.nn.functional as F
from Rep import Rep

class Fourier_2D(nn.Module):
    # 声明一个类变量，用于在多线程/多实例间共享数据
    _shared_last_lamd = None

    def __init__(self, lamd=0.8, lamd_min=0.5, lamd_max=0.95, D0=32, eps=1e-6) -> None:
        super(Fourier_2D, self).__init__()
        print("\n🚀")
        self.lamd_min = lamd_min
        self.lamd_max = lamd_max
        self.D0 = D0
        self.eps = eps

        # 防溢出保护与初始偏置计算
        lamd = max(min(lamd, lamd_max - 1e-4), lamd_min + 1e-4)
        init_norm = (lamd - lamd_min) / (lamd_max - lamd_min)
        init_bias = math.log(init_norm / (1.0 - init_norm))

        # 注册可学习参数
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

    # 使用 property 装饰器，将类变量伪装成实例属性，供外部读取
    @property
    def last_lamd(self):
        return Fourier_2D._shared_last_lamd

    def forward(self, input):
        b, c, n, h, w = input.shape
        dtype, device = input.dtype, input.device

        freq = torch.fft.fftshift(
            torch.fft.fftn(input.float(), dim=(3, 4), norm='ortho'),
            dim=(3, 4)
        )

        lp_F, hp_F, boundary_F = self._CircleFilter(h, w, device, freq.real.dtype)

        power = freq.abs().pow(2)

        low_energy = (power * lp_F).sum(dim=(2, 3, 4), keepdim=True)
        high_energy = (power * hp_F).sum(dim=(2, 3, 4), keepdim=True)

        low_num = lp_F.sum() * n + self.eps
        high_num = hp_F.sum() * n + self.eps

        low_energy = low_energy / low_num
        high_energy = high_energy / high_num

        # 截断特征能量比值的梯度回传，保证 alpha 和 beta 的优化仅受下游任务主梯度驱动
        low_ratio = low_energy / (low_energy + high_energy + self.eps)
        low_ratio = low_ratio.detach()

        # Sigmoid 硬约束自适应计算
        lamd_norm = torch.sigmoid(self.alpha * (low_ratio - 0.5) + self.beta)
        lamd = self.lamd_min + (self.lamd_max - self.lamd_min) * lamd_norm

        # 将计算结果更新到全局唯一的类变量中，确保能够被外部正确读取
        Fourier_2D._shared_last_lamd = lamd.detach().mean().item()

        adaptive_filter = lp_F * lamd + hp_F * (1.0 - lamd)
        adaptive_filter = adaptive_filter + boundary_F * (1.0 / 3.0)

        freq = freq * adaptive_filter

        output = torch.fft.ifftn(
            torch.fft.ifftshift(freq, dim=(3, 4)),
            s=(h, w),
            dim=(3, 4),
            norm='ortho'
        ).real

        return output.to(dtype=dtype)

    def _CircleFilter(self, H, W, device, dtype):
        x_center, y_center = int(H // 2), int(W // 2)

        X, Y = torch.meshgrid(
            torch.arange(0, H, device=device),
            torch.arange(0, W, device=device),
            indexing='ij'
        )

        circle = torch.sqrt(
            (X.float() - x_center) ** 2 + (Y.float() - y_center) ** 2
        )

        r = max(1, min(H, W) // self.D0)

        lp_F = (circle < r).to(dtype)
        hp_F = (circle > r).to(dtype)
        boundary_F = (circle == r).to(dtype)

        lp_F = lp_F.view(1, 1, 1, H, W)
        hp_F = hp_F.view(1, 1, 1, H, W)
        boundary_F = boundary_F.view(1, 1, 1, H, W)

        return lp_F, hp_F, boundary_F


try:
    from thop import profile
except ImportError:
    profile = None

try:
    from pytorch_memlab import MemReporter
except ImportError:
    MemReporter = None


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, expand_ratio):
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
            nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def create_depthwise_layer(channel):
        return nn.Sequential(
            nn.Conv3d(
                channel,
                channel,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                groups=channel,
                bias=False
            ),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def create_project_layer(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False
            ),
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