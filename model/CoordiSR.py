# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.weight_conv =nn.Conv2d(G0+2,3,3,1,1)

    def forward(self, input ,scale=2):
        input_sub = self.sub_mean(input)
        f__1 = self.SFENet1(input_sub)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        # Up-sampling net
        up_x, up_input = upsampling(x,input_sub,scale)
        w_x = self.weight_conv(up_x)

        x = up_input + w_x

        x = self.add_mean(x)
        return x


def upsampling(x, input,scale=2):
    ###here is our meta learning upsampling function
    # the scale
    h, w = x.size()[2:]
    hr_h, hr_w = int(scale * h), int(scale * w)


    pos_h = torch.arange(0,hr_h/scale,1/scale).repeat(hr_w)
    pos_h = pos_h.view(1,1,hr_w,hr_h).transpose(3,2)

    pos_w = torch.arange(0,hr_w/scale, 1/scale).repeat(hr_h)
    pos_w = pos_w.view(1,1,hr_h,hr_w)

    pos = torch.cat((pos_h,pos_w),1)

    int_pos = torch.floor(pos)
    res_pos = pos - int_pos
    res_pos = res_pos.cuda()
    res_pos = res_pos.expand(x.size(0),res_pos.size(1),hr_h,hr_w)

    up_x = nn.functional.upsample(x,[hr_h,hr_w], mode='bilinear')
    up_input = nn.functional.upsample(input,[hr_h,hr_w], mode='bilinear')
    #print(up_x.size())
    #print(res_pos.size())
    pos_x = torch.cat((up_x,res_pos),1)

    return pos_x,up_input


