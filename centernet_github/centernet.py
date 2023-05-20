from resnet import ResNet
from decoder import Decoder
from head import Head
from fpn import FPN
import sys
import os
import torch
from torch import nn
import torch.nn.functional as F

parent_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_folder_path)


def map2coords(h, w, stride):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


class CenterNetPyTorch(nn.Module):
    def __init__(self, cfg):
        super(CenterNetPyTorch, self).__init__()
        self.backbone = ResNet(in_c=cfg.in_c, slug=cfg.slug)
        if cfg.fpn:
            self.fpn = FPN(self.backbone.outplanes)
        self.upsample = Decoder(self.backbone.outplanes if not cfg.fpn else 2048, cfg.bn_momentum)
        self.head = Head(channel=cfg.head_channel, num_classes=cfg.num_classes)

        self._fpn = cfg.fpn

    def forward(self, x):
        feats = self.backbone(x)
        if self._fpn:
            feat = self.fpn(feats)
        else:
            feat = feats[-1]
        return self.head(self.upsample(feat))
    

if __name__ == '__main__':
    from config import general_cfg
    # from ..config import config
    import pdb

    model = CenterNetPyTorch(cfg=general_cfg.centernet_pytorch)
    input = torch.rand(1, 15, 512, 512)
    hm, om = model(input)
    pdb.set_trace()
    print(hm.shape)