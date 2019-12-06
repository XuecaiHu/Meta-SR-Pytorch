from model import common
import math
import torch.nn as nn
import torch

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return EDSR(args, dilated.dilated_conv)
    else:
        return EDSR(args)



class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.args = args
        self.scale_idx = 0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))



        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        ## position to weight

        self.P2W = Pos2Weight(inC=args.n_feats)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)
    
    def repeat_weight(self, weight, scale, inw,inh):
        k = int(math.sqrt(weight.size(0)))
        outw  =inw * scale
        outh = inh * scale
        weight = weight.view(k, k, -1)
        scale_w = (outw+k-1) // k
        scale_h = (outh + k - 1) // k
        weight = torch.cat([weight] * scale_h, 0)
        weight = torch.cat([weight] * scale_w, 1)

        weight = weight[0:outh,0:outw,:]

        return weight
    def forward(self, x, pos_mat):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        up_x = self.repeat_x(res)     ### the output is (N*r*r,inC,inH,inW)

        # N*r^2 x [inC * kH * kW] x [inH * inW]
        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)
        local_weight = self.repeat_weight(local_weight,scale_int,x.size(2),x.size(3))
        
        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out = self.add_mean(out)
        ###
        return out




    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale[scale_idx]
