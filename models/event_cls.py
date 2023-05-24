from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.model_utils import *
    

class EventClassifier(pl.LightningModule):
    def __init__(self, general_cfg, model_cfg) -> None:
        super(EventClassifier, self).__init__()
        self.general_cfg = general_cfg
        self.config = model_cfg
        in_c = model_cfg.in_c   # input will have shape n x 5 x 128 x 128
        self.layers = nn.Sequential(
            ConvBlock(in_c, in_c*3, act=nn.SiLU()),
            ConvBlock(in_c*3, in_c*3, act=nn.SiLU()),
            ConvBlock(in_c*3, in_c*3, act=nn.SiLU()),

            nn.Conv2d(in_c*3, in_c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Dropout(p=0.1),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=in_c, out_features=in_c),
            nn.SiLU(),
            nn.Linear(in_features=in_c, out_features=model_cfg.n_class),     # n_class = 2
        )

        self.criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([self.config.bounce_weight, self.config.net_weight]))
        self._reset_metrics()

    
    def forward(self, input):
        x = self.layers(input)
        return x
    
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.general_cfg.training.base_lr, weight_decay=self.general_cfg.training.weight_decay)
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
                'monitor': 'val_ev_acc',
                'frequency': 1,
                'interval': 'epoch'
            }
        }


    def _reset_metrics(self):
        self.running_ev_true, self.running_ev_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        self.val_running_ev_true, self.val_running_ev_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)


    def common_step(self, batch, batch_idx):
        inputs, ev_true = batch
        ev_pred = self.forward(inputs)
        loss = self.criterion(ev_pred, ev_true)

        # compute metrics
        diff = torch.abs(torch.sigmoid(ev_pred) - ev_true)   # shape nx2, event_pred is not sigmoided inside the model
        n_true = (diff < self.general_cfg.decode.ev_diff_thresh).all(dim=1).sum()

        return loss, n_true
    

    def training_step(self, batch, batch_idx):
        inputs, ev_true = batch
        loss, n_true = self.common_step(batch, batch_idx)
        self.running_ev_true += n_true
        self.running_ev_total += ev_true.shape[0]

        self.log_dict({
            'train_loss': loss,
            'train_ev_acc': self.running_ev_true / self.running_ev_total
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        inputs, ev_true = batch
        loss, n_true = self.common_step(batch, batch_idx)
        self.val_running_ev_true += n_true
        self.val_running_ev_total += ev_true.shape[0]

        self.log_dict({
            'val_loss': loss,
            'val_ev_acc': self.val_running_ev_true / self.val_running_ev_total
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def on_train_epoch_start(self) -> None:
        print('\n')
        self._reset_metrics()
    
    def on_train_epoch_end(self):
        self._reset_metrics()

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def on_validation_epoch_end(self) -> None:
        self._reset_metrics()



if __name__ == '__main__':
    model = EventClassifier()


