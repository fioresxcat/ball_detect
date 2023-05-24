import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class CustomFocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, pos_weight=0.4, eps=1e-5):
        """
            set alpha = 0 and we have standard BCE loss
            set pos_weight < 0.5 will penalty false positive more 
        """
        super(CustomFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.neg_weight = 1 - self.pos_weight
        self.eps = eps


    def forward(self, ypred: torch.tensor, ytrue: torch.tensor):
        """
            Y_true: tensor, shape n x 128 x 128, 
            y_pred: tensor, shape n x 1 x 128 x 128 
        """
        ypred = torch.squeeze(ypred, dim=1)
        class_weight_mask = torch.where(ytrue!=0, torch.tensor(self.pos_weight), torch.tensor(self.neg_weight))
        
        loss1 = ytrue.eq(1).float() * (1-ypred)**self.alpha * torch.log(ypred+self.eps)
        loss2 = ytrue.ne(1).float() * (1-ytrue)**self.beta * ypred**self.alpha * torch.log(1-ypred+self.eps)
        loss = loss1 + loss2
        loss = class_weight_mask * loss
        return -1 / ypred.shape[0] / ypred.shape[1] / ypred.shape[2] * loss.sum()



class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''
        target = torch.unsqueeze(target, dim=1)
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class MyLoss(nn.Module):
    def __init__(self, pos_weight) -> None:
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, ypred, ytrue):
        ypred = ypred.squeeze(dim=1)
        diff = ypred - ytrue
        diff[diff<0] *= self.pos_weight
        return (diff**2).mean()


class WBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, ypred, ytrue):
        ypred = ypred.squeeze(dim=1)
        return -(
            ((1-ypred)**2) * ytrue *
            torch.log(torch.clamp(ypred, min=1e-15, max=1)) +
            ypred**2 * (1-ytrue) *
            torch.log(torch.clamp(1-ypred, min=1e-15, max=1))
        ).mean()
    

class ModifiedFocalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, gt):
        '''
        Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt (batch x c x h x w)
        '''
        gt = gt.unsqueeze(dim=1)
        
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        pred = torch.clamp(pred, 1e-4, 1-1e-4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class CenterNetLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, pos_weight=0.4, l_off=1):
        super(CenterNetLoss, self).__init__()
        # self.focal_loss = CustomFocalLoss(alpha=alpha, beta=beta, pos_weight=pos_weight)
        self.focal_loss = ModifiedFocalLoss()
        self.l_off = l_off      # offset loss weight


    def forward(self, hm_pred, om_pred, hm_true, om_true, out_pos):
        keypoint_loss = self.focal_loss(hm_pred, hm_true)
        offset_loss = 0
        for i in range(len(hm_pred[0])):
            single_om_pred = om_pred[i]
            single_om_true = om_true[i]
            single_pos = out_pos[i]
            if (single_pos == -100).all():   # frame ko có bóng => ko tinhs l1 loss
                continue
            offset_loss += torch.abs(single_om_pred[:, single_pos[1], single_pos[0]] - single_om_true[:, single_pos[1], single_pos[0]]).sum()
        return keypoint_loss + offset_loss*self.l_off



class CenterNetEventLoss(nn.Module):
    def __init__(self, only_bounce=True, bounce_pos_weight=0.7, l_ball=1, l_event=1, bounce_weight=1,  net_weight=1):
        super(CenterNetEventLoss, self).__init__()
        self.focal_loss = ModifiedFocalLoss()
        self.l_ball = l_ball      # offset loss weight
        self.l_event = l_event
        self.loss_weight = torch.tensor([bounce_weight, net_weight])
        self.bounce_pos_weight = bounce_pos_weight      # use when only bounce
        self.only_bounce = only_bounce

    def forward(self, hm_pred, om_pred, ev_pred, hm_true, om_true, ev_true, out_pos):
        keypoint_loss, offset_loss = 0, 0
        if self.l_ball > 0:
            # keypoint loss
            keypoint_loss = self.focal_loss(hm_pred, hm_true)

            # offset loss
            offset_loss = 0
            for i in range(len(hm_pred[0])):
                single_om_pred = om_pred[i]
                single_om_true = om_true[i]
                single_pos = out_pos[i]
                if (single_pos == -100).all():   # frame ko có bóng => ko tinhs l1 loss
                    continue
                offset_loss += torch.abs(single_om_pred[:, single_pos[1], single_pos[0]] - single_om_true[:, single_pos[1], single_pos[0]]).sum()

        # event loss
        if self.only_bounce:
            ev_loss = F.binary_cross_entropy_with_logits(
                ev_pred[:, 0],   #  0 = class bounce
                ev_true[:, 0],
                weight=1 - torch.abs(self.bounce_pos_weight-ev_true[:, 0])
            )
        else:
            ev_loss = F.binary_cross_entropy_with_logits(
                ev_pred, 
                ev_true, 
                weight=self.loss_weight
            )

        ball_loss = self.l_ball * (keypoint_loss + offset_loss)
        ev_loss = self.l_event * ev_loss
        loss = ball_loss + ev_loss
        return loss, ball_loss, ev_loss


# class CenterNetEventOnlyBounceLoss(nn.Module):
#     def __init__(self, pos_weight=0.4, l_ball=1, l_event=1, bounce_weight=1, net_weight=3):
#         super(CenterNetEventOnlyBounceLoss, self).__init__()
#         self.focal_loss = ModifiedFocalLoss()
#         self.l_ball = l_ball      # offset loss weight
#         self.l_event = l_event


#     def forward(self, hm_pred, om_pred, ev_pred, hm_true, om_true, ev_true, out_pos):
#         keypoint_loss = self.focal_loss(hm_pred, hm_true)

#         offset_loss = 0
#         for i in range(len(hm_pred[0])):
#             single_om_pred = om_pred[i]
#             single_om_true = om_true[i]
#             single_pos = out_pos[i]
#             if (single_pos == -100).all():   # frame ko có bóng => ko tinhs l1 loss
#                 continue
#             offset_loss += torch.abs(single_om_pred[:, single_pos[1], single_pos[0]] - single_om_true[:, single_pos[1], single_pos[0]]).sum()
#         # pdb.set_trace()
#         if ev_true
#         ev_loss = F.binary_cross_entropy_with_logits(ev_pred, ev_true) * self.bounce_weight
#         return self.l_ball * (keypoint_loss + offset_loss) + self.l_event * ev_loss


if __name__ == '__main__':
    batch_hm_true = torch.rand(2, 128, 128)
    hm = torch.rand(2, 1, 128, 128)

    batch_om_true = torch.rand(2, 2, 128, 128)
    om_pred = torch.rand(2, 2, 128, 128)

    abs_pos = torch.tensor([
        [50, 50],
        [50, 50]
    ])
    criterion = CenterNetLoss()
    loss = criterion(hm, om_pred, batch_hm_true, batch_om_true, abs_pos)
    print(loss)