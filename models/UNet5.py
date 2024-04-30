from audioop import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from torch.distributions.normal import Normal
from torchvision import transforms
from models.losses import ContrastiveLoss

from pathlib import Path
if __name__ == "__main__":
    import losses as losses
else:
    import models.losses as losses

class DoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, 
        conv_op=nn.Conv3d,conv_args=None,
        norm_op=nn.BatchNorm3d, norm_op_args=None,
        drop_op=nn.Dropout3d, drop_op_args=None,
        non_line_op=nn.LeakyReLU, non_line_op_args=None,
        is_conv_pool=False,
        is_id_mapping=False):
        super(DoubleConv, self).__init__()
        
        self.drop_op = drop_op
        self.is_id_mapping = is_id_mapping
        
        if conv_args is None:
            conv_args = {'kernel_size':3,'stride':1,'padding':1,'dilation': 1, 'bias': True}
        if norm_op_args is None:
            norm_op_args = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if drop_op_args is None:
            drop_op_args = {'p': 0.5, 'inplace': True}
        if non_line_op_args is None:
            non_line_op_args = {'negative_slope': 1e-2, 'inplace': True}
        if is_conv_pool:
            conv_args2 = {'kernel_size':3,'stride':2,'padding':1,'dilation': 1, 'bias': True}
        else:
            conv_args2 = conv_args

        self.conv1 = conv_op(in_channel, mid_channel, **conv_args)
        self.norm1 = norm_op(mid_channel, **norm_op_args)
        self.non_line1 = non_line_op(**non_line_op_args)
        
        self.conv2 = conv_op(mid_channel, out_channel, **conv_args2)
        self.norm2 = norm_op(out_channel, **norm_op_args)
        self.non_line2 = non_line_op(**non_line_op_args)
        
        if drop_op is not None:
            self.drop1 = drop_op(**drop_op_args)
            self.drop2 = drop_op(**drop_op_args)
        
        if self.is_id_mapping:
            self.id_mapping = nn.Conv3d(in_channel,out_channel,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        if self.drop_op is not None:
            x1 = self.non_line1(self.norm1(self.drop1(self.conv1(x))))
            x2 = self.non_line2(self.norm2(self.drop2(self.conv2(x1))))
        else:
            x1 = self.non_line1(self.norm1(self.conv1(x)))
            x2 = self.non_line2(self.norm2(self.conv2(x1)))
        if self.is_id_mapping:
            x_id = self.id_mapping(x)
            x2 = x2 + x_id
        return x2


class Encoder(nn.Module):
    def __init__(self, stage_num=5, conv_pool=False, pool_args=None, moment=0.96):
        super(Encoder,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        
        # 网络参数
        self.stage_num = stage_num
        self.conv_pool = conv_pool
        self.moment = moment
        self.encode_blocks = []
        self.decode_blocks = []
        self.down_pool = []
        self.contra_loss = ContrastiveLoss(1)

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        
        self.encode_blocks = [
            DoubleConv(1,32,32,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(32,64,64,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(64,128,128,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(128,256,256,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(256,512,512,drop_op=None,is_conv_pool=conv_pool),
        ]
        if self.conv_pool is False:
            self.down_pool = [
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
            ]

        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.down_pool = nn.ModuleList(self.down_pool)
        
    def forward(self, moving, fixed, moving_type, fixed_type):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = moving
        y = fixed
        x_type = moving_type
        y_type = fixed_type
        bs = x.size(0)
        self.prototype_loss = torch.tensor(0)
        self.contra = torch.tensor(0)

        skip_moving = []
        skip_fixed = []
        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            if self.conv_pool is False and d!=self.stage_num-1:
                x = self.down_pool[d](x)
                y = self.down_pool[d](y)
            skip_moving.append(x)
            skip_fixed.append(y)

            for b  in range(bs):
                self.setPrototype(x[b].detach(),x_type[b],d)
                self.setPrototype(y[b].detach(),y_type[b],d)
                # Center_loss = Center_loss + losses.Center_loss(y[i], y_type[i], self.prototypes[d]) + losses.Center_loss(x[i], x_type[i], self.prototypes[d])
                if d >= 1:
                    self.prototype_loss = self.prototype_loss + self.proto_loss(self.gram(y[b:b+1]), self.gram(self.prototypes[d][y_type[b]].unsqueeze(0))) \
                                                            + self.proto_loss(self.gram(x[b:b+1]), self.gram(self.prototypes[d][x_type[b]].unsqueeze(0)))
                    self.contra = self.contra + self.contra_loss(self.gram(x[b:b+1]), self.gram(y[b:b+1]), 1)
        # self.prototype_loss = self.prototype_loss/self.stage_num
        
        return skip_moving, skip_fixed
    
    def forward1(self, warped_x):
        warped_list = []
        for d in range(self.stage_num):
            warped_x = self.encode_blocks[d](warped_x)
            if self.conv_pool is False and d!=self.stage_num-1:
                warped_x = self.down_pool[d](warped_x)
            warped_list.append(warped_x)
        return warped_list

    def setPrototype(self,x,x_filename,stage):
        old_shots = self.pro_dict[stage].get(x_filename,0)
        if old_shots == 0:
            self.prototypes[stage][x_filename] = x
        
        p = self.prototypes[stage][x_filename]
        new_shots = old_shots + 1
        # prototype = (x + p*old_shots) / new_shots
        prototype = (1-self.moment) * x + self.moment * p

        self.pro_dict[stage][x_filename] = new_shots
        self.prototypes[stage][x_filename] = prototype
    
    def getPrototypes_sim(self, target, stage):
        proto_ft = torch.stack(self.prototpyes[stage]).flatten(2)
        target_flat = torch.flatten(target,2)
        sp_sum = torch.sum(target_flat.unsqueeze(1) * proto_ft.unsqueeze(0), 2)
        sp_sum = sp_sum.mean(2)
        index = torch.argmax(sp_sum)
        return self.prototypes[stage][list(self.keys())[index]]
    
    def getPrototypes(self,x_filename,stage):
        prototypes = []
        for filename in x_filename:
            prototypes.append(self.prototypes[stage][filename])
        return torch.stack(prototypes)

    def cosdist(self,fts,prototype,scaler=20):
        dist = F.cosine_similarity(fts, prototype, dim=1) * scaler
        return dist.detach().cpu().numpy()
    
    def gram(self, X):
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        return X.bmm(X.transpose(1,2))
    
    def gram2(self, X):
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        gram = X.bmm(X.transpose(1,2))
        factor = X.square().sum(-1,keepdims=True).sqrt()  # b, c
        norm = factor.bmm(factor.transpose(1,2))
        return gram / norm
    
    def proto_loss(self, x, proto):
        c, n = x.shape[-2], x.shape[-1]
        proto_loss = torch.square((x-proto)).sum()
        proto_loss = proto_loss / (c*c*n*n)
        return proto_loss


class Decoder(nn.Module):
    def __init__(self, stage_num=5, up_sample=False, up_sample_args=None):
        super(Decoder,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        
        # 网络参数
        self.stage_num = stage_num
        self.up_sample = up_sample
        self.encode_blocks = []
        self.decode_blocks = []
        self.down_pool = []
        self.up_sample = []

        if up_sample and up_sample_args is None:
            up_sample_args = {'size':2, 'scale_factor':2, 'mode':'nearest', 'align_corners':None}
        
        self.decode_blocks = [
            DoubleConv(512*2,512,256,drop_op=None),
            DoubleConv(256*3,512,128,drop_op=None),
            DoubleConv(128*3,256,64,drop_op=None),
            DoubleConv(64*3,128,32,drop_op=None),
            DoubleConv(32*3,64,16,drop_op=None)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d(3*256,3*256,2,2),
                nn.ConvTranspose3d(3*128,3*128,2,2),
                nn.ConvTranspose3d(3*64,3*64,2,2),
                nn.ConvTranspose3d(3*32,3*32,2,2),
            ]

        self.decode_blocks = nn.ModuleList(self.decode_blocks)
        self.up_sample = nn.ModuleList(self.up_sample)
        conv_fn = getattr(nn, 'Conv%dd' % 3)
        self.flow = conv_fn(16, 3, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # self.batch_norm = getattr(nn, "BatchNorm{0}d".format(3))(3)

        self.last_conv = conv_fn(16, 1, kernel_size=3, padding=1)

        
    def forward(self, skip_m, skip_f, m_type, f_type, prototypes):
        
        con_list = skip_f[:2]
        for i in range(2,len(skip_f)):
            con_list.append(torch.stack([prototypes[i][f_t] for f_t in f_type], dim=0))

        # 第一次上采样不使用跳链接
        d_x = self.decode_blocks[0](torch.concat([skip_m[-1], con_list[-1]],dim=1))
        # best_p = self.getPrototypes(f_type,-1)
        # x = self.decode_blocks[0](torch.concat([x, best_p],dim=1))
        
        # 上采样
        # for u in range(1,self.stage_num):
        #     if self.up_sample:
        #         x = self.up_sample[u](x)
        #     else:
        #         x = self.up_trans[u](x)
        #     x = torch.concat([skip_m[-u],x,self.getPrototypes(f_type,-(u+1)), skip_f[-u]], dim=1)
        #     x = self.decode_blocks[u](x)
        
        for u in range(1,self.stage_num):
            # d_x = torch.concat([d_x, skip_m[-(u+1)], skip_f[-(u+1)]], dim=1)
            d_x = torch.concat([d_x, skip_m[-(u+1)], con_list[-(u+1)]], dim=1)
            d_x = self.decode_blocks[u](self.up_sample[u-1](d_x))

        # flow = self.flow(d_x)
        return self.last_conv(d_x)

class Dis(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self,):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        input_nc=1
        ndf=64
        n_layers=3
        norm_layer=nn.InstanceNorm3d

        super(Dis, self).__init__()
        use_bias = (norm_layer != nn.BatchNorm2d)

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
    def gen_loss(self, fake):
        out_fake_list = self.forward(fake)
        loss = 0
        for _, (out_fake) in enumerate(out_fake_list):
            loss += torch.mean((out_fake - 1)**2)
        return loss

    def dis_loss(self, fake, true):
        out_fake_list = self.forward(fake)
        out_true_list = self.forward(true)
        loss = 0
        for _, (out_fake, out_true) in enumerate(zip(out_fake_list, out_true_list)):
            loss += torch.mean((out_fake - 0)**2) + torch.mean((out_true - 1)**2)
        return loss


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0                             
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MHAttention(nn.Module):
    def __init__(self, type_num, dim):
        super(MHAttention, self).__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(type_num,dim,1)))

    def forward(self, X, fixed):
        # X,fixed: (B,C,Z,H,W), (C=dim)
        b,c,z,h,w = X.size()
        f_flat = torch.flatten(fixed,2) # B,C,ZHW
        f_T = f_flat.transpose(1,2) # B,ZHW,C
        fw = torch.matmul(f_T.unsqueeze(1), self.W)  #(B,1,ZHW,C) *(type_n,C,1) = (B,type_n,ZHW,1) 确定每个像素属于的类型
        fw_sm = fw.softmax(dim=1) # (B,type_n,ZHW,1)
        
        x_flat = torch.flatten(X,2)
        x_T = x_flat.transpose(1,2).unsqueeze(1) # B,1,ZHW,C
        fw_x = torch.mul(fw_sm,x_T) # (B,type_n,ZHW,C)
        fw_x = torch.sum(fw_x,dim=1)
        fw_x = fw_x.transpose(1,2).reshape((b,c,z,h,w))
        return fw_x


class SpatitalAttention(nn.Module):
    def __init__(self, dim, liner_dim=512, ndims=3, win_width=None):
        super(SpatitalAttention, self).__init__()
        liner_dim = dim # 特征维度不变
        self.q_w = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(liner_dim,dim)))
        self.k_w = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(liner_dim,dim)))
        self.v_w = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(liner_dim,dim)))

         # set window size
        win1 = [1] * ndims
        win2 = [9] * ndims if win_width is None else win_width
        
        # compute filters
        self.sum_filt1 = torch.ones([liner_dim, 1, *win1])
        self.sum_filt2 = torch.ones([1, 1, *win2]).to("cuda")
        self.sum_filt3 = torch.ones([liner_dim, 1, *win2])

        pad_no = math.floor(win2[0]/2)

        if ndims == 1:
            self.stride = (1)
            self.padding = (pad_no)
        elif ndims == 2:
            self.stride = (1,1)
            self.padding = (pad_no, pad_no)
        else:
            self.stride = (1,1,1)
            self.padding = (pad_no, pad_no, pad_no)

        # get convolution function
        self.conv_fn = getattr(F, 'conv%dd' % ndims)


    def forward(self, moving, fixed):
        # moving,fixed: (B,C,Z,H,W)
        b,c,z,h,w = moving.size()
        moving_flat = torch.flatten(moving,2)  # B,C,ZHW
        fixed_flat = torch.flatten(fixed,2)

        q = torch.matmul(moving_flat,self.q_w) # B,liner_dim,ZHW
        k = torch.matmul(fixed_flat,self.k_w)
        v = torch.matmul(fixed_flat,self.v_w)

        qk = (q*k).reshape(b,self.linear_dim,z,h,w) # B,liner_dim,Z,H,W
        qk = self.conv_fn(qk, self.sum_filt1, stride=1, padding=0) # B,1,Z,H,W
        
        qkv  = qk * v.reshape(b,self.linear_dim,z,h,w) # B,liner_dim,Z,H,W
        
        qk_sum = self.conv_fn(qk, self.sum_filt2, stride=self.stride, padding=self.padding) # B,1,Z,H,W
        qkv_sum = self.conv_fn(qkv, self.sum_filt3, stride=self.stride, padding=self.padding) # B,liner_dim,Z,H,W
        norm_qkv = qkv_sum / qk_sum  # B,liner_dim,Z,H,W

        return norm_qkv


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        # src B,C,Z,Y,Z
        # flow 3,z,y,x
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # return F.grid_sample(src, new_locs, mode=self.mode)
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)


# def main():
#     model = UNet(1,2)
#     x1 = torch.rand(2,1,64,192,224)
#     x2 = torch.rand(2,1,64,192,224)
#     x_file = Path("D:\\dataset\\kidney_2021-01-06_WKQ\\output_nii\\xiamen_nii_results_resize\\CMP\\HUANG_CHUNSHUI_XM21023682.nii.gz")
#     # x_filename = x_filename.split("P\\")[0]
#     # x_filename = x_filename.split("ze\\")[1]
#     x_filename = x_file.parent.stem
#     with torch.no_grad():
#         o = model(x1,x_filename)
#         o = model(x2,'CMP')
#     print(o.size())

# if __name__ == "__main__":
#     main()