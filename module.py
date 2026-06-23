import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from Rep import Rep
from Fourier import AFFBlock

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
