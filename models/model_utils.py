import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import cv2
import numpy as np
import pdb


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1, act=nn.Identity()):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_c)
        self.act = act
    
    def forward(self, input):
        return self.act(self.bn(self.conv(input)))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = ConvBlock(in_c, out_c, stride=stride, act=nn.ReLU())
        self.conv2 = ConvBlock(out_c, out_c)
        if stride > 1:
            self.downsample = ConvBlock(in_c, out_c, stride=stride)
        
    def forward(self, input):
        x = self.conv1(input)
        x = F.dropout(x, p=0.1)
        x = self.conv2(x)
        if self.stride > 1:
            identity = self.downsample(input)
        else:
            identity = input
        x = x + identity
        return F.relu(x)
    

class DownSampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res1 = ResBlock(in_c, out_c, stride=2)
        self.dropout = nn.Dropout(p=0.1)
        self.res2 = ResBlock(out_c, out_c)
    
    def forward(self, input):
        return self.res2(self.dropout(self.res1(input)))


class UpSampleBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.res1 = ResBlock(in_c, in_c)
        self.res2 = ResBlock(in_c, in_c)
        self.conv = ConvBlock(in_c, out_c, act=nn.ReLU())

    def forward(self, input):
        x = F.dropout(self.res1(input), p=0.1)
        x = F.dropout(self.res2(x), p=0.1)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        x = self.conv(x)
        return x


class ConnectBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res = ResBlock(in_c=in_c, out_c=out_c)

    def forward(self, input):
        x = self.res(input)
        x = F.dropout(x, p=0.1)
        return x
    

class HourGlassModule(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self._init_layers()
    

    def _init_layers(self):
        self.encoder = nn.ModuleList([
            DownSampleBlock(in_c=self.in_c, out_c=128),    # /8 so với input size to the whole model
            DownSampleBlock(in_c=128, out_c=128),   # /16
            DownSampleBlock(in_c=128, out_c=196),   # /32
            DownSampleBlock(in_c=196, out_c=256),   # /64
        ])
        self.decoder = nn.ModuleList([
            UpSampleBlock(in_c=256, out_c=196),
            UpSampleBlock(in_c=196, out_c=128),
            UpSampleBlock(in_c=128, out_c=128),
            UpSampleBlock(in_c=128, out_c=128),
        ])
        self.last_conv_path = nn.ModuleList([
            ConvBlock(in_c=128, out_c=96, act=nn.ReLU()),
            ConvBlock(in_c=96, out_c=96, act=nn.ReLU()),
            ResBlock(in_c=96, out_c=self.out_c)
        ])
        self.connect_block = nn.ModuleList([
            ConvBlock(in_c=self.in_c, out_c=96, act=nn.ReLU()),
            ConvBlock(in_c=self.in_c, out_c=128, act=nn.ReLU()),
            ConnectBlock(in_c=128, out_c=128),
            ConnectBlock(in_c=128, out_c=128),
            ConnectBlock(in_c=196, out_c=196),
            ConnectBlock(in_c=256, out_c=256),
        ])

    
    def forward(self, input):
        e1 = self.encoder[0](input)
        e2 = self.encoder[1](e1)
        e3 = self.encoder[2](e2)
        e4 = self.encoder[3](e3)

        d1 = self.decoder[0](self.connect_block[5](e4))
        # pdb.set_trace()
        d1 = d1 + self.connect_block[4](e3)

        d2 = self.decoder[1](d1)
        d2 = d2 + self.connect_block[3](e2)

        d3 = self.decoder[2](d2)
        d3 = d3 + self.connect_block[2](e1)

        d4 = self.decoder[3](d3)
        d4 = d4 + self.connect_block[1](input)

        x = self.last_conv_path[0](d4)
        x = self.last_conv_path[1](x) + self.connect_block[0](input)
        x = self.last_conv_path[2](x)

        return x



def find_max_index(hm):
    """
        hm shape: n x 1 x out_h x out_w
    """
    batch, cat, height, width = hm.size()
    topk_scores, topk_inds = torch.topk(hm.view(batch, cat, -1), 1)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    pos = torch.concat([topk_xs, topk_ys], dim=1).squeeze(dim=-1)
    return pos      # shape n x 2


def decode_hm(hm, kernel, conf_thresh):
    """
        hm shape: n x 1 x out_w x out_h
        if there is a heatmap in the batch with no detected ball, will return ball_pos as [0, 0]
    """
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == hm).float()
    hm = hm * keep   # n x 1 x out_h x out_w
    hm = torch.where(hm<conf_thresh, 0, hm)
    pos = find_max_index(hm)
    return pos


def decode_hm(hm, kernel, conf_thresh):
    """
        hm shape: n x 1 x out_w x out_h
        if there is a heatmap in the batch with no detected ball, will return ball_pos as [0, 0]
    """
    hm = hm.squeeze(dim=1)
    hm = torch.where(hm<conf_thresh, 0, hm)
    max_indices = torch.argmax(hm.view(hm.shape[0], -1), dim=1)
    pos = torch.stack([max_indices // hm.shape[2], max_indices % hm.shape[2]], dim=1)
    pos[:, [0, 1]] = pos[:, [1, 0]]
    return pos


def decode_hm_by_contour(batch_hm, conf_thresh):
    """
        hm shape: n x 1 x out_w x out_h
        om shape: n x 2 x out_w x out_h
        if there is a heatmap in the batch with no detected ball, will return ball_pos as [0, 0]

    """
    batch_hm = batch_hm.squeeze(dim=1).cpu().numpy()
    batch_hm_int = (batch_hm*255).astype(np.uint8)
    batch_pos = []
    for idx, hm_int in enumerate(batch_hm_int):
        hm = batch_hm[idx]
        ret, binary_hm = cv2.threshold(hm_int, conf_thresh*255, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary_hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            xmin, ymin, w, h = cv2.boundingRect(largest_contour)
            max_idx = np.unravel_index(np.argmax(hm[ymin:ymin+h, xmin:xmin+w]), (h, w))
            ball_x, ball_y = (max_idx[1]+xmin, max_idx[0] + ymin)
            batch_pos.append(torch.tensor([ball_x, ball_y]))
        else:
            batch_pos.append(torch.tensor([0, 0]))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_pos = torch.stack(batch_pos, dim=0).to(device)
    return batch_pos



def compute_metrics(hm_pred, hm_true, kernel, conf_thresh, rmse_thresh):
    """
        hm_pred: shape n x 1 x out_h x out_w
        hm_true: shape n x out_h x out_w
    """
    pos_pred = decode_hm(hm_pred, kernel, conf_thresh)
    pos_true = find_max_index(hm_true.unsqueeze(1))
    diff = torch.sqrt(torch.pow(pos_pred-pos_true, 2).sum(dim=1))
    n_true = (diff<rmse_thresh).int().sum()
    rmse = diff.sum()
    return n_true, rmse
