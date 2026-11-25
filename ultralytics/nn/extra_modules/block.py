import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f, RepC3
from ..modules.conv import Conv, autopad
from .rep_block import DiverseBranchBlock


class SCGM(nn.Module):
    def __init__(self, c1, kernel_size=3, gate_scale=0.05):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        self.gain = nn.Parameter(torch.zeros(1))

        self.gate_scale = gate_scale

    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)

        g = self.pool(g)

        g = g - g.mean(dim=(2, 3), keepdim=True)

        g = torch.tanh(g)

        out = x * (1.0 + self.gate_scale * self.gain * g)
        return out


class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=7, v_kernel_size=7):
        super().__init__()
        self.h_conv = nn.Conv2d(
            ch, ch,
            kernel_size=(1, h_kernel_size),
            stride=1,
            padding=(0, h_kernel_size // 2),
            groups=ch,
            bias=False
        )
        self.v_conv = nn.Conv2d(
            ch, ch,
            kernel_size=(v_kernel_size, 1),
            stride=1,
            padding=(v_kernel_size // 2, 0),
            groups=ch,
            bias=False
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.v_conv(self.h_conv(x)))
        return x * attn_factor


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class EAHAB(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)
        self.caa_attention = CAA(ch=inc * 2)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.caa_attention(torch.cat(out, 1)))


class CSP_EAM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EAHAB(self.c, [3, 6, 9, 12]) for _ in range(n))


class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class CA_DBBC3(RepC3):
    def __init__(self, c1, c2, n=3, e=1):
        super().__init__(c1, c2, n, e)
        c_ = int(c2 * e)

        layers = [DiverseBranchBlock(c_, c_, 3) for _ in range(n)]
        layers.append(CoordAtt(c_, c_))
        self.m = nn.Sequential(*layers)


class FGlo(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CGNet(nn.Module):
    def __init__(self, nIn, dilation_rate=2, reduction=16):
        super().__init__()
        nOut = 2 * nIn
        self.conv1x1 = Conv(nIn, nOut, 3, s=2)

        self.F_loc = nn.Conv2d(nOut, nOut, 3, padding=1, groups=nOut)
        self.F_sur = nn.Conv2d(nOut, nOut, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=nOut)

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = Conv.default_act
        self.reduce = Conv(2 * nOut, nOut,1,1)

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur],1)
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)

        output = self.F_glo(joi_feat)

        return output


class _PooledConv(Conv):

    def forward(self, x):
        if self.bn.training and x.shape[0] == 1:
            bn = self.bn
            x = F.batch_norm(self.conv(x), bn.running_mean, bn.running_var, bn.weight, bn.bias,
                             training=False, eps=bn.eps)
            return self.act(x)
        return super().forward(x)


class SDFM(nn.Module):

    def __init__(self, channels=64, r=4):
        super(SDFM, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _PooledConv(2 * channels, 2 * inter_channels),
            _PooledConv(2 * inter_channels, 2 * channels, act=nn.Sigmoid()),
        )

        self.channel_agg = Conv(2 * channels, channels)

        self.local_att = nn.Sequential(
            Conv(channels, inter_channels, 1),
            Conv(inter_channels, channels, 1, act=False),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _PooledConv(channels, inter_channels, 1),
            _PooledConv(inter_channels, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x1, x2 = data
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input)
        local_w = self.local_att(agg_input)
        global_w = self.global_att(agg_input)
        w = self.sigmoid(local_w * global_w)
        xo = w * x1 + (1 - w) * x2
        return xo


StemSafeGate = SCGM
CSP_MutilScaleEdgeInformationEnhance = CSP_EAM
MutilScaleEdgeInformationEnhance = EAHAB
DBBC3 = CA_DBBC3
ContextGuidedBlock_Down = CGNet
