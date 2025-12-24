import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window


# 计算 ssim 损失函数
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2


    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

def mse(img1, img2, window_size=9):
    max_val = 1
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res


# 方差计算
def std(img,  window_size=3):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

# def sum(img,  window_size=9):

#     padd = window_size // 2
#     (_, channel, height, width) = img.size()
#     window = create_window(window_size, channel=channel).to(img.device)
#     win1 = torch.ones_like(window)
#     res = F.conv2d(img, win1, padding=padd, groups=channel)
#     return res



def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    # std_ir = std(img_ir)
    # std_vi = std(img_vis)
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    # m = torch.mean(img_ir)
    # w_ir = torch.where(img_ir > m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map1 = map1
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    map2 = map2

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()

def final_mse(img_ir, img_vis, img_fuse):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    w_vi = torch.where(img_ir <= m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    res = map1 * mse_ir + map2 * mse_vi
    res = res * w_vi
    return res.mean()




def final_l1_loss(img_ir, img_vis, img_fuse):
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    map_ir = std_ir / (std_ir + std_vi + 0.00001)
    map_vi = std_vi / (std_ir + std_vi + 0.00001)
    return F.l1_loss(map_ir * img_fuse, map_ir * img_ir) + F.l1_loss(map_vi * img_fuse, map_vi * img_vis)


def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss


class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)


import torch.nn.functional as F
def cos_sim(vi_same, vi_diff, ir_same, ir_diff):
    same = torch.cosine_similarity(vi_same, ir_same)
    diff = torch.cosine_similarity(vi_diff, ir_diff)
    ones = torch.ones_like(same)
    zeros = torch.zeros_like(diff)
    return F.l1_loss(same, ones)

class PearSon_corr(nn.Module):
    def __init__(self):
        super(PearSon_corr, self).__init__()

    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

def Pearson_loss(vi_same, vi_diff, ir_same, ir_diff):
    cc = PearSon_corr()
    detial = cc(vi_same, ir_same)**2
    base = torch.abs((cc(vi_diff, ir_diff)+0.00001)) #防止除0
    return torch.div(detial, base)

from models.common import gradient
def gard_max_loss(vis, inf, f):
    return F.l1_loss(gradient(f), torch.max(gradient(vis), gradient(inf)))



########VGG_loss
from torchvision.models import vgg19
import torch
import torch.nn.functional as F
from torchvision import models

class vgg_loss(torch.nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()
        vgg_model = vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:].cuda()
        vgg_model.eval()
        for param in vgg_model.parameters():
            param.requires_grad = False  # 使得之后计算梯度时不进行反向传播及权重更新
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        # self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            # print("vgg_layers name:",name,module)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        #print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        ###########这里的目的是将输入张量处理为channel=3
        output_ = output
        output = torch.cat((output_, output_), dim=1)
        output = torch.cat((output_, output), dim=1)
        #####################
        gt_ = gt
        gt = torch.cat((gt_, gt_), dim=1)
        gt = torch.cat((gt_, gt), dim=1)
        ###############
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(
                zip(output_features, gt_features, self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature) * loss_weight)
        return sum(loss)  # /len(loss)



