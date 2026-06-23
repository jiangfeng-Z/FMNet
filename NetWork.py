import torch, copy, math
import torch.nn as nn
import torch.nn.functional as F
from Fourier import *
from Rep import Rep
from module import *

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
            Feature_Extraction(base)
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
