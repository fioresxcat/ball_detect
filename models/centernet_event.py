import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from .model_utils import *
from .basemodel import BaseModel
from loss import CenterNetEventOnlyBounceLoss
import torchvision
from centernet_yolo.model import CenterNetYolov8EventOnlyBounce, IHeadEventOnlyBounce



class CenterNetYoloEventOnlyBounce(BaseModel):
    def __init__(self, general_cfg, model_cfg):
        super(CenterNetYoloEventOnlyBounce, self).__init__(general_cfg)
        self.config = model_cfg
        self._init_layers()
        self._init_lightning_stuff()
    

    def _init_layers(self):
        self.model = CenterNetYolov8EventOnlyBounce(
            version=self.config.version,
            nc=1,
            load_pretrained_yolov8= False if self.general_cfg.training.prev_ckpt_path is not None else self.config.load_pretrained_yolov8 
        )
        self.model.backbone.backbone[0].conv = nn.Conv2d(self.config.in_c, self.model.backbone.backbone[0].conv.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if self.config.freeze_event:
            self.model.head.event_spot.requires_grad_(False)

        if self.config.freeze_ball:
            self.model.backbone.requires_grad_(False)
            self.model.neck.requires_grad_(False)
            for name, param in self.model.head.named_parameters():
                if 'event_spot' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def _init_lightning_stuff(self):
        self.criterion = CenterNetEventOnlyBounceLoss(
            l_event = 0 if self.config.freeze_event else self.config.l_event,
            l_ball = 0 if self.config.freeze_ball else self.config.l_ball,
        )
        self._reset_metric()


    def configure_optimizers(self):
        base_lr = self.general_cfg.training.base_lr
        opt = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=self.general_cfg.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='max',
            factor=0.2,
            patience=10,
            verbose=True
        )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.general_cfg.training.monitor,
                'frequency': 1,
                'interval': 'epoch'
            }
        }
    

    def _reset_metric(self):
        self.running_ball_true, self.running_ball_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        self.val_running_ball_true, self.val_running_ball_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)

        self.running_ev_true, self.running_ev_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        self.val_running_ev_true, self.val_running_ev_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)

        self.running_rmse = torch.tensor(0, device=self.device)
        self.val_running_rmse = torch.tensor(0, device=self.device)


    def common_step(self, batch, batch_idx):
        imgs, hm_true, om_true, out_pos, normalized_pos, event_true = batch
        hm_pred, om_pred, event_pred = self.forward(imgs)
        loss = self.criterion(hm_pred, om_pred, event_pred, hm_true, om_true, event_true, out_pos)
        return hm_pred, om_pred, event_pred, loss
    

    def training_step(self, batch, batch_idx):
        imgs, hm_true, om_true, out_pos, normalized_pos, event_true = batch
        hm_pred, om_pred, event_pred, loss = self.common_step(batch, batch_idx)

        # compute ball detect metrics
        n_true, rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, rmse_thresh=2)
        self.running_ball_true += n_true
        self.running_ball_total += hm_pred.shape[0]

        # compute event metric
        if not self.config.freeze_event:
            diff = torch.abs(torch.sigmoid(event_pred) - event_true)   # shape nx2, event_pred is not sigmoided inside the model
            n_true = (diff < self.general_cfg.decode.ev_diff_thresh).all(dim=1).sum()
            self.running_ev_true += n_true
            self.running_ev_total += hm_pred.shape[0]

        self.log_dict({
            'train_loss': loss,
            'train_acc': torch.round(self.running_ball_true/self.running_ball_total, decimals=3),
            'train_rmse': rmse,
            'train_ev_acc': torch.round(self.running_ev_true/self.running_ev_total, decimals=3) if not self.config.freeze_event else -1,
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        imgs, hm_true, om_true, out_pos, normalized_pos, event_true = batch
        hm_pred, om_pred, event_pred, loss = self.common_step(batch, batch_idx)

        # compute ball detect metrics
        n_true, rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, rmse_thresh=2)
        self.val_running_ball_true += n_true
        self.val_running_ball_total += hm_pred.shape[0]

        # compute event metric
        if not self.config.freeze_event:
            diff = torch.abs(torch.sigmoid(event_pred) - event_true)   # shape nx2
            n_true = (diff < self.general_cfg.decode.ev_diff_thresh).all(dim=1).sum()
            self.val_running_ev_true += n_true
            self.val_running_ev_total += hm_pred.shape[0]

        self.log_dict({
            'val_loss': loss,
            'val_acc': torch.round(self.val_running_ball_true/self.val_running_ball_total, decimals=3),
            'val_rmse': rmse,
            'val_ev_acc': torch.round(self.val_running_ev_true/self.val_running_ev_total, decimals=3) if not self.config.freeze_event else -1,
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def on_train_epoch_end(self):
        self._reset_metric()

    def on_train_epoch_start(self) -> None:
        print('\n')
        self._reset_metric()
    
    def on_validation_epoch_end(self) -> None:
        self._reset_metric()

    def on_validation_epoch_start(self) -> None:
        self._reset_metric()