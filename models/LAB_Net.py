#MCAL-Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from timm.models.layers import trunc_normal_
from models.LCFNet_model import Regress_net
from utils.util import color_trans


class Volume_2D(nn.Module):
    def __init__(self, indim=256):
        super(Volume_2D, self).__init__()
        self.c2f_dim = 33
        self.conv_dim = indim
        coarse_range = torch.arange(-1., 1. + 0.01, step=0.0625).reshape(1, -1)  # 1 k
        self.color_range = nn.Parameter(coarse_range[None, :, :, None, None], requires_grad=False)  # 1 1 k 1 1
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=1, stride=1, padding=0),
            nn.Hardswish(inplace=True),
            nn.Conv2d(self.conv_dim, 2 * self.c2f_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        cost_feature = self.coarse_conv(x).view(b, 2, self.c2f_dim, h, w)
        prob = F.softmax(cost_feature, dim=2)  # b 2 k h w
        exp = torch.sum(prob * self.color_range, dim=2)  # b 2 h w
        return exp


class ColorCompenateNet(nn.Module):
    def __init__(self, cont_dim=64, color_dim=64):
        super(ColorCompenateNet, self).__init__()
        self.color_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1)
        )
        self.context_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1)
        )
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1)
        )
        self.hsv_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1)
        )
        self.encoder_0 = nn.Sequential(
            nn.InstanceNorm2d(256, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/2
        )
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(256, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/4
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(512, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/8
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(512, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/16
        )
        self.color_decoder3 = Volume_2D(indim=512)
        self.color_decoder2 = Volume_2D(indim=514)
        self.color_decoder1 = Volume_2D(indim=258)
        self.color_decoder0 = Volume_2D(indim=258)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, lab, rgb, hsv):
        # step1: feature extraction
        h, w = lab.shape[2:]
        l = lab[:, :1, :, :]
        ab = lab[:, 1:, :, :]
        feat_color = self.color_encoder(ab)
        feat_cont = self.context_encoder(l)
        feat_rgb = self.rgb_encoder(rgb)
        feat_hsv = self.hsv_encoder(hsv)
        feat = torch.cat([feat_color, feat_cont, feat_rgb, feat_hsv], dim=1) # 256
        feat0 = self.encoder_0(feat)  # 1/2 256
        h0, w0 = feat0.shape[2:]
        feat1 = self.encoder_1(feat0)  # 1/4 256
        h1, w1 = feat1.shape[2:]
        feat2 = self.encoder_2(feat1)  # 1/8 512
        h2, w2 = feat2.shape[2:]
        feat3 = self.encoder_3(feat2)  # 1/16 512

        # step2: multi-scale probabilistic volumetric fusion
        pre_ab3 = self.color_decoder3(feat3)
        pre_ab3 = F.interpolate(pre_ab3, size=(h2, w2), mode='bilinear', align_corners=True)  # 1/8 2
        feat2 = torch.cat([feat2, pre_ab3], dim=1)  # 1/8 514

        pre_ab2 = self.color_decoder2(feat2)
        pre_ab2 = F.interpolate(pre_ab2, size=(h1, w1), mode='bilinear', align_corners=True)  # 1/4 2
        feat1 = torch.cat([feat1, pre_ab2], dim=1)  # 1/4 258

        pre_ab1 = self.color_decoder1(feat1)
        pre_ab1 = F.interpolate(pre_ab1, size=(h0, w0), mode='bilinear', align_corners=True)  # 1/2 2
        feat0 = torch.cat([feat0, pre_ab1], dim=1)  # 1/2 258

        pre_ab0 = self.color_decoder0(feat0)
        pre_ab0 = F.interpolate(pre_ab0, size=(h, w), mode='bilinear', align_corners=True)  # 1 2

        return l, pre_ab0, pre_ab1, pre_ab2, pre_ab3


class WaterNet(nn.Module):
    def __init__(self, dim=64, init_weights=False):
        super(WaterNet, self).__init__()
        self.color = ColorCompenateNet(cont_dim=dim, color_dim=dim)
        regress_net = Regress_net(num_classes=2, init_weights=init_weights)
        self.L_predict = regress_net
        self.trans = color_trans()

    def forward(self, lab, rgb, hsv, gray,  params_origin):
        l, ab0, ab1, ab2, ab3 = self.color(lab, rgb, hsv)
        params_l = self.L_predict(l, gray) #b*2
        trans_l = self.trans.trans_l(l*100., params_origin, params_l)
        final_l = trans_l * (1 + gray)
        lab = torch.cat([final_l, ab0 * 127.], dim=1)
        rgb = kornia.color.lab_to_rgb(lab)  # 0~1
        return {'ab_pred0': ab0, 'ab_pred1': ab1, 'ab_pred2': ab2, 'ab_pred3': ab3, 'l': final_l/100., 'lab': lab, 'lab_rgb': rgb, 'params_l': params_l}
