import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

class Head(nn.Module):
    def __init__(self, c1, c, nc=20) -> None:
        super().__init__()
        '''
                        --> out_hm
        x --> x --> x -- --> out_wh 
                        --> out_reg
        '''
        self.conv1 = Conv(c1, c*4, k=3, s=1)
        self.conv2 = Conv(c*4, c*2, k=3, s=1)
        self.conv1 = Conv(c*2, c, k=3, s=1)

        self.hm_out = nn.Sequential(
            Conv(c, c, 3, 1),
            nn.Conv2d(c, nc, 1),
            nn.Sigmoid()
        )

        self.wh_out = nn.Sequential(
            Conv(c, c, 3, 1),
            nn.Conv2d(c, 2, 1),
        )

        self.reg_out = nn.Sequential(
            Conv(c, c, 3, 1),
            nn.Conv2d(c, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(self.conv2(self.conv3(x)))
        
        out_hm = self.hm_out(x)
        out_wh = self.wh_out(x)
        out_reg = self.reg_out(x)

        return out_hm, out_wh, out_reg

                
def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3)) # N, WxH, C
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections

class Decoder(nn.Module):
    def __init__(self, max_boxes):
        super(Decoder, self).__init__()
        self.ctdet_decode = ctdet_decode
        self.max_boxes = max_boxes

    def forward(self, out):
        return self.ctdet_decode(out[0], out[1], out[2], K=self.max_boxes)

# ----------------------------- #
## Differentiable Binarization ##
# ----------------------------- #

class ScaleChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        print(self.avgpool)
        self.fc1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.fc2 = nn.Conv2d(out_planes, num_features, 1, bias=False)
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = F.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        global_x = F.softmax(global_x, 1)
        return global_x
    

class ScaleChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelSpatialAttention, self).__init__()
        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes , 1, bias=False),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, bias=False)
        )
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # global_x = self.avgpool(x)
        #shape Nx4x1x1
        global_x = self.channel_wise(x).sigmoid()
        #shape: NxCxHxW
        global_x = global_x + x
        #shape:Nx1xHxW
        x = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x) + global_x
        global_x = self.attention_wise(global_x)
        return global_x

class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid() 
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels, inter_channels , out_features_num=4, attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels=in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        if self.type == 'scale_spatial':
            self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4, out_features_num)
        elif self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels//2, out_features_num)


    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features_num
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])
        return torch.cat(x, dim=1)



import contextlib
import math
import torch
import torch.nn as nn
import yaml
import pdb
import os
import sys

# parent_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
# sys.path.append(parent_folder_path)
# sys.path.append(os.getcwd())


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch, version='n'):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
    depth, width = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple'))
    depth, width, max_channels = scales[version]

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']+d['head'][:-1]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        # if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
        #          BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
        if m in (Conv, Bottleneck, SPPF, C2f, nn.ConvTranspose2d):    
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            # if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x):
            if m in (C2f,):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class Backbone(nn.Module):
    def __init__(self, version='s', load_pretrained=True) -> None:
        super().__init__()

        with open('yolov8.yaml') as f:
            d = yaml.safe_load(f)
        self.backbone, self.save = parse_model(d, 3, version)
        if load_pretrained:
            try: 
                v8_pretrained = os.path.join(DIR, 'yolov8%s.pt'%version)
                self.backbone.load_state_dict(torch.load(v8_pretrained, map_location='cpu'), strict=False)
                print('Load successfully yolov8%s backbone weights !'%version)
            except:
                print('Cannot load yolov8%s backbone weights !'%version)

    def forward(self, inp):
        out_bb = {}
        x = inp

        for i in range(22):
            if i not in [11, 14, 17, 20]:
                x = self.backbone[i](x)
            
                if i in [4, 6, 9, 12, 15, 18, 21]:
                    out_bb[i] = x
            elif i == 11:
                x = self.backbone[i]((x, out_bb[6]))
            elif i == 14:
                x = self.backbone[i]((x, out_bb[4]))
            elif i == 17:
                x = self.backbone[i]((x, out_bb[12]))
            elif i == 20:
                x = self.backbone[i]((x, out_bb[9]))
        
        del out_bb[4]
        del out_bb[6]
        del out_bb[9]
        del out_bb[12]
                # 1/8     1/16         1/32      
        return out_bb[15], out_bb[18], out_bb[21]

class Neck(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512],
                 inner_channels=192):
        super().__init__()
        c = inner_channels // 3
        
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.in1 = Conv(in_channels[2], c, k=1) #for layer21
        self.in2 = Conv(in_channels[1], c, k=1) #for layer18
        self.in3 = Conv(in_channels[0], c, k=1) #for layer15

        self.concat_attention = ScaleFeatureSelection(inner_channels, c, 3, attention_type='scale_channel_spatial')
        
        # self.ia = ImplicitA(c*3)
        # self.ims = nn.ModuleList([ImplicitM(c) for i in range(3)])

    def forward(self, features):
        out15, out18, out21 = features
        up1 = self.in1(self.up1(out21))
        up2 = self.in2(self.up2(out18))
        up3 = self.in3(self.up3(out15))
        
        # fuse = torch.cat((self.ims[0](up1), self.ims[1](up2), self.ims[2](up3)), 1)
        fuse = torch.cat((up3, up2, up1), 1)
        fuse = self.concat_attention(fuse, [up3, up2, up1])
        # fuse = self.ims[0](up1) + self.ims[1](up2) + self.ims[2](up3)
        # fuse = self.ia(fuse)
        return fuse

class IHead(nn.Module):
    def __init__(self, c, nc=1) -> None:
        super().__init__()
        # self.ia, self.im = ImplicitA(c), ImplicitM(c)
        self.conv1 = Conv(c*3, c*3, k=3, s=1)
        self.conv2 = Conv(c*3, c*2, k=3, s=1)
        self.conv3 = Conv(c*2, c, k=3, s=1)

        self.hm_out = nn.Sequential(
            Conv(c, c, 3, 1), #self.ia,
            Conv(c, c, 3, 1), #self.im,
            nn.Conv2d(c, nc, 1, bias=True),
            nn.Sigmoid()
        )
        self.hm_out[-2].bias.data.fill_(-4.6)

        self.reg_out = nn.Sequential(
            Conv(c, c, 3, 1), #self.ia,
            Conv(c, c, 3, 1), #self.im,
            nn.Conv2d(c, 2, 1),
        )


    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        out_hm = self.hm_out(x)
        out_reg = self.reg_out(x)
        return out_hm, out_reg
    

class IHeadEvent(IHead):
    def __init__(self, c, nc=1) -> None:
        super(IHeadEvent, self).__init__(c, nc)
        
        # self.event_spot = nn.Sequential(
        #     Conv(c*3, c*3, k=3, s=1),
        #     Conv(c*3, c*2, k=3, s=1),
        #     Conv(c*2, c, k=3, s=1),

        #     Conv(c, c, k=3, s=1),
        #     nn.Dropout(p=0.1),
        #     Conv(c, c, k=3, s=1),
        #     nn.Dropout(p=0.1),
        #     Conv(c, c, k=3, s=1),
        #     nn.Dropout(p=0.1),

        #     nn.AdaptiveAvgPool2d(output_size=1),
        #     nn.Flatten(),
        #     nn.Linear(in_features=c, out_features=c),
        #     nn.SiLU(),
        #     nn.Linear(in_features=c, out_features=2),
        #     # nn.Sigmoid()          # excluding sigmoid to train with BCEWithLogits Loss. At inference will add a sigmoid
        # )

        self.event_spot = nn.Sequential(
            Conv(c*3, c*3, k=3, s=2, p=1),  # / 2
            Conv(c*3, c*3, k=3, s=2, p=1),   # / 2
            Conv(c*3, c, k=3, s=2, p=1), # /2

            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),   # /2, shape 64 x 4 x 4
            nn.SiLU(),
            nn.Dropout2d(p=0.1),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=1024, out_features=512),
            nn.SiLU(),
            nn.Linear(in_features=512, out_features=2),
            # nn.Sigmoid()          # excluding sigmoid to train with BCEWithLogits Loss. At inference will add a sigmoid
        )

    def forward(self, input):
        x = self.conv3(self.conv2(self.conv1(input)))
        out_hm = self.hm_out(x)
        out_reg = self.reg_out(x)
        out_event = self.event_spot(input)

        return out_hm, out_reg, out_event



class IHeadEventOnlyBounce(IHead):
    def __init__(self, c, nc=1) -> None:
        """
            input has shape n x 16 x 128 x 128
        """
        super(IHeadEventOnlyBounce, self).__init__(c, nc)
        self.event_spot = nn.Sequential(
            Conv(c*3, c*3, k=3, s=2, p=1),  # / 2
            Conv(c*3, c*3, k=3, s=2, p=1),   # / 2
            Conv(c*3, c, k=3, s=2, p=1), # /2

            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),  # /2
            nn.Dropout2d(p=0.1),
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),   # /2, shape 64 x 4 x 4
            nn.Dropout2d(p=0.1),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=1024, out_features=512),
            nn.SiLU(),
            nn.Linear(in_features=512, out_features=1),
            # nn.Sigmoid()          # excluding sigmoid to train with BCEWithLogits Loss. At inference will add a sigmoid
        )

    def forward(self, input):
        x = self.conv3(self.conv2(self.conv1(input)))
        out_hm = self.hm_out(x)
        out_reg = self.reg_out(x)
        out_event = self.event_spot(input)

        return out_hm, out_reg, out_event
    

class CenterNetYolov8(nn.Module):
    def __init__(self, version='n', nc=1, load_pretrained_yolov8=False):
        super().__init__()
        self.version = version
        scales = dict(
            # [depth, width, max_channels]
            n= [0.33, 0.25, 1024],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
            s= [0.33, 0.50, 1024],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
            m= [0.67, 0.75, 768],   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
            l= [1.00, 1.00, 512],   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
            x= [1.00, 1.25, 512],   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
        )
        gd, gw, max_channels = scales[version]
        ch = [make_divisible(c_*gw, 8) for c_ in [64, 128, 256, 512, max_channels]] #16, 32, 64, 128, 256
        self.ch = ch

        self.backbone = Backbone(version, load_pretrained=load_pretrained_yolov8)
        inner_channels = ch[2]*3
        self.neck = Neck(ch[2:], inner_channels)
        self.head = IHead(ch[2], nc)
        

    def forward(self, inp):
        features = self.backbone(inp)
        fuse = self.neck(features)
        out = self.head(fuse)
        return out


    def fuse(self):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.
        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward

        return self
    
    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.
        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.
        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model
    


class CenterNetYolov8Event(CenterNetYolov8):
    def __init__(self, version='n', nc=1, load_pretrained_yolov8=False):
        super(CenterNetYolov8Event,self).__init__(version, nc, load_pretrained_yolov8)
        self.head = IHeadEvent(self.ch[2], nc)


class CenterNetYolov8EventOnlyBounce(CenterNetYolov8):
    def __init__(self, version='n', nc=1, load_pretrained_yolov8=False):
        super(CenterNetYolov8EventOnlyBounce, self).__init__(version, nc, load_pretrained_yolov8)
        self.head = IHeadEventOnlyBounce(self.ch[2], nc)

    


if __name__ == '__main__':
    import time
    import numpy as np
    from models.unet import EffSmpUnet
    import json
    from easydict import EasyDict as edict
    from ultralytics import YOLO

    #------------------------ yolov8 ----------------------------
    # model = CenterNetYolov8()
    model = 
    # model = YOLO('yolov8n.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    x = torch.rand(1, 3, 640, 640, device=device)

    ls_time = []

    for i in range(100):
        out = model(x)

    for i in range(1000):
        start = time.perf_counter()
        out = model(x)
        cur_time = time.perf_counter()-start
        ls_time.append(cur_time)
        print('time: ', cur_time)
    mean_time = np.mean(ls_time)
    print(mean_time)

    # # ------------------------ unet ----------------------------
    # general_cfg = edict(json.load(open('ckpt/exp46_effsmpunet_silu_all_head_nho/general_cfg.json')))
    # general_cfg.data.n_input_frames = 1
    # model_cfg = edict(json.load(open('ckpt/exp46_effsmpunet_silu_all_head_nho/model_cfg.json')))
    # model_cfg.in_c = 3
    # model = EffSmpUnet(general_cfg=general_cfg, model_cfg=model_cfg)
    # model.eval().to('cuda')
    # x = torch.rand(1, 3, 640, 640, device='cuda')

    # for i in range(100):
    #     out = model(x)

    # ls_time = []
    # for i in range(1000):
    #     start = time.perf_counter()
    #     out = model(x)
    #     cur_time = time.perf_counter()-start
    #     ls_time.append(cur_time)
    #     print('time: ', cur_time)
    # mean_time = np.mean(ls_time)
    # print(mean_time)

