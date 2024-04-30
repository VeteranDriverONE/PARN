import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim2(trans, fixed, moving, window, window_size, channel, size_average = True):
    mu_t = F.conv2d(trans, window, padding = window_size//2, groups = channel)
    mu_f = F.conv2d(fixed, window, padding = window_size//2, groups = channel)
    mu_m = F.conv2d(moving, window, padding = window_size//2, groups = channel)

    mu_t_sq = mu_t.pow(2)
    mu_f_sq = mu_f.pow(2)
    mu_m_sq = mu_m.pow(2)

    mu_t_mu_f = mu_t * mu_f
    mu_t_mu_m = mu_t * mu_m

    sigma_t_sq = torch.clamp(F.conv2d(trans*trans, window, padding = window_size//2, groups = channel) - mu_t_sq, min=0)
    sigma_f_sq = torch.clamp(F.conv2d(fixed*fixed, window, padding = window_size//2, groups = channel) - mu_f_sq, min=0)
    sigma_tf = F.conv2d(trans*fixed, window, padding = window_size//2, groups = channel) - mu_t_mu_f

    sigma_m_sq = torch.clamp(F.conv2d(moving*moving, window, padding = window_size//2, groups = channel) - mu_m_sq, min=0)
    sigma_tm = F.conv2d(trans*moving, window, padding = window_size//2, groups = channel) - mu_t_mu_m

    C1 = 0.01**2
    C2 = 0.03**2
    C3 = 0.5 * C2

    # ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    lum_con_map = (2*mu_t*mu_f+C1)/(mu_t_sq + mu_f_sq + C1) + (sigma_t_sq.sqrt() * sigma_f_sq.sqrt() +C2)/(sigma_t_sq + sigma_f_sq + C2)
    
    luminance = (2*mu_t_mu_f+C1)/(mu_t_sq + mu_f_sq+C1)
    
    contrast = (2*sigma_t_sq.sqrt()*sigma_f_sq.sqrt() + C2) / (sigma_t_sq + sigma_f_sq + C2)
    
    structure = (sigma_tm + C3)/(sigma_t_sq.sqrt() * sigma_m_sq.sqrt() +C3)

    if size_average:
        return  luminance.mean(), contrast.mean(), structure.mean()
    else:
        return luminance.mean(1).mean(1), contrast.mean(1).mean(1), structure.mean(1).mean(1)
    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    def ssim2(self, trans, fixed, moving):
        (_, channel, _, _) = trans.size()

        if channel == self.channel and self.window.data.type() == trans.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if trans.is_cuda:
                window = window.cuda(trans.get_device())
            window = window.type_as(trans)
            
            self.window = window
            self.channel = channel

        return _ssim2(trans, fixed, moving, window, self.window_size, channel, self.size_average)


