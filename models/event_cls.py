from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.model_utils import *
    

class EventClassifier(pl.LightningModule):
    def __init__(self, model_cfg) -> None:
        super(EventClassifier, self).__init__()
        self.config = model_cfg
        in_c = model_cfg.in_c
        self.layers = nn.Sequential(
            ConvBlock(in_c*3, in_c*3),
            ConvBlock(in_c*3, in_c*2),
            ConvBlock(in_c*2, in_c),

            ConvBlock(in_c, in_c),
            nn.Dropout(p=0.1),
            ConvBlock(in_c, in_c),
            nn.Dropout(p=0.1),
            ConvBlock(in_c, in_c),
            nn.Dropout(p=0.1),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=in_c, out_features=in_c),
            nn.SiLU(),
            nn.Linear(in_features=in_c, out_features=model_cfg.n_class),
        )

        self.criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([self.config.bounce_weight, self.config.net_weight]))
    
    
    def forward(self, input):
        x = self.layers(input)
        return x
    
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.base_lr, weight_decay=self.general_cfg.training.weight_decay)
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


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        out = self.forward(inputs)
        loss = self.criterion(out, labels)



    def on_train_epoch_start(self) -> None:
        print('\n')
    

