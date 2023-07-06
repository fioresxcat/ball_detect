from typing import Any, Dict
import torch
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self, general_cfg):
        super(BaseModel, self).__init__()
        self.general_cfg = general_cfg


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
                'monitor': 'val_acc',
                'frequency': 1,
                'interval': 'epoch'
            }
        }
    

    def on_train_epoch_end(self):
        self._reset_manual_metric()

    def on_train_epoch_start(self) -> None:
        print('\n')
        self._reset_manual_metric()
    
    def on_validation_epoch_end(self) -> None:
        self._reset_manual_metric()

    def on_validation_epoch_start(self) -> None:
        self._reset_manual_metric()
