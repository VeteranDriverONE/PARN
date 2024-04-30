import torch
import torch.nn.functional as F
import numpy as np
import math
from operator import itemgetter


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


# class Dice:
#     """
#     N-D dice for segmentation
#     """

#     def loss(self, y_true, y_pred):
#         ndims = len(list(y_pred.size())) - 2
#         vol_axes = list(range(2, ndims + 2))
#         top = 2 * (y_true * y_pred).sum(dim=vol_axes)
#         bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
#         dice = torch.mean(top / bottom)
#         return -dice

def reg_loss(fixed, flows):
    # sim_loss = pearson_correlation(fixed, warped)
    # Regularize all flows
    if len(fixed.size()) == 4: #(N, C, H, W)
        reg_loss = sum([regularize_loss(flow) for flow in flows])
    else:
        reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
    
    return  reg_loss 

def reg_loss2(fixed, flows):
    # sim_loss = pearson_correlation(fixed, warped)
    # Regularize all flows
    if len(fixed.size()) == 4: #(N, C, H, W)
        reg_loss = [regularize_loss(flow) for flow in flows]
    else:
        reg_loss = [regularize_loss_3d(flow) for flow in flows]
    
    return  reg_loss 

def reg_loss1(flow):
    return flow.flatten(1).pow(2).mean(1).mean()

def regularize_loss_3d(flow):
    """
    flow has shape (batch, 3, 512, 521, 512)
    """
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    d = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

    return d / 3.0

def regularize_loss(flow):
    """
    flow has shape (batch, 2, 521, 512)
    """
    dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
    dy = (flow[..., 1:] - flow[..., :-1]) ** 2

    d = torch.mean(dx) + torch.mean(dy)

    return d / 2.0

def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed)
    mean2 = torch.mean(flatten_warped)
    var1 = torch.mean((flatten_fixed - mean1) ** 2)
    var2 = torch.mean((flatten_warped - mean2) ** 2)

    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2))
    eps = 1e-6
    pearson_r = cov12 / torch.sqrt((var1 + eps) * (var2 + eps))

    raw_loss = 1 - pearson_r

    return raw_loss
 

class NCE:
    def __init__(self, types_name):
        return 

    def loss(self, source_ft, source_type, prototypes):
        
        proto = list(prototypes.values()) 
        proto_ft = torch.stack(proto).flatten(2)  # type_num,C,ZHW
        if len(source_type)>1:
            proto_ft_b = torch.stack(itemgetter(*source_type)(prototypes)).flatten(2)
        else:
            proto_ft_b = itemgetter(*source_type)(prototypes).unsqueeze(0).flatten(2)
        # proto_ft = torch.index_select(proto, 0, source_type)  # B,C,ZHW

        source_ft = torch.flatten(source_ft,2)  # B,C,ZHW
        sp = torch.nn.functional.cosine_similarity(source_ft, proto_ft_b, 1)  # B,ZHW
        sp = sp.mean(1) # B

        source_m = torch.sqrt(torch.sum(source_ft.pow(2),1,keepdim=True))  # 向量模, B,1,ZHW
        proto_m = torch.sqrt(torch.sum(proto_ft.pow(2),1,keepdim=True))  # 向量模, type_num,1,ZHW
        # sp1=torch.sum((source_ft / source_m)* (proto_ft/proto_m), 1)
        source_ft_m = source_ft / source_m  # 除以模， B,C,ZHW
        proto_ft_m = proto_ft/ proto_m  # 除以模， type_num,C,ZHW

        sp_sum = torch.sum(source_ft_m.unsqueeze(1) * proto_ft_m.unsqueeze(0), 2)  # B,1,C,ZHW * 1,type_num,C,ZHW = B,type_num,ZHW
        sp_sum = sp_sum.mean(2).sum(1)

        loss = sp / sp_sum
        return -loss.mean()


def Center_loss(source_ft,source_file,source_pros):
    """
    center loss 
    source_fts: feature of the moving image,
    source_pro: now prototypes
    """
    source_pro = source_pros[source_file]
    center_loss = torch.mean(torch.square((source_ft-source_pro)))

    return center_loss

def Center_loss2(source_ft,source_file,source_pros):
    """
    center loss 
    source_fts: feature of the moving image,
    source_pro: now prototypes
    """
    source_pro = source_pros[source_file]
    c,z,w,h = source_ft.shape
    return (1-torch.cosine_similarity(source_ft.reshape(c,z*h*w),source_pro.reshape(c,z*h*w),dim=0)).sum()

def triplet_loss(source_ft,source_file,prototypes,scalar):
    dif = 0
    for k,v in prototypes:
        if k==source_file:
            same = torch.sqrt(source_ft-v)
        else:
            dif+=torch.sqrt(source_ft-v)
    triplrt_loss = same + dif
    return triplrt_loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        c = output1.shape[1]
        # output1和output2为两个向量，label表示两向量是否是同一类，同一类为0,不同类为1
        euclidean_distance = F.pairwise_distance(output1, output2)/c
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive

