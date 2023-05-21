import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from .model_utils import *
from .basemodel import BaseModel
from loss import ModifiedFocalLoss, CustomFocalLoss
import torchvision
import segmentation_models_pytorch as smp

criterion_dict = {
    'modified_focal_loss': ModifiedFocalLoss,
    # 'custom_focal_loss': CustomFocalLoss
}


class BaseUnetModel(BaseModel):
    def __init__(self, general_cfg, model_cfg):
        super().__init__(general_cfg)
        self.config = model_cfg
        self._init_lightning_stuff()    


    def _init_lightning_stuff(self):
        self.criterion = criterion_dict[self.config.loss]()
        self.running_true, self.running_total = 0, 0
        self.val_running_true, self.val_running_total = 0, 0
        self.running_rmse = 0
        self.val_running_rmse = 0

    
    def configure_optimizers(self):
        base_lr = self.general_cfg.training.base_lr
        opt = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=self.general_cfg.training.weight_decay)
        
        def lr_foo(ep):
            if ep < 1:
                return 10
            else:
                return 1
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_foo)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
            }
        }


    def common_step(self, batch, batch_idx):
        imgs, hm_true, om_true, out_pos, norm_pos = batch
        hm_pred = self.forward(imgs)
        loss = self.criterion(hm_pred, hm_true)
        return hm_pred, hm_true, loss
    

    def training_step(self, batch, batch_idx):
        hm_pred, hm_true, loss = self.common_step(batch, batch_idx)
        n_true, sum_batch_rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, self.general_cfg.decode.rmse_thresh)
        self.running_true += n_true
        self.running_total += hm_pred.shape[0]
        self.running_rmse += sum_batch_rmse
        self.log_dict({
            'train_loss': loss,
            'train_acc': round(self.running_true/self.running_total, 3),
            'train_rmse': round(self.running_rmse/self.running_total, 3),
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        hm_pred, hm_true, loss = self.common_step(batch, batch_idx)
        n_true, sum_batch_rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, self.general_cfg.decode.rmse_thresh)
        self.val_running_true += n_true
        self.val_running_total += hm_pred.shape[0]
        self.val_running_rmse += sum_batch_rmse

        self.log_dict({
            'val_loss': loss,
            'val_acc': round(self.val_running_true/self.val_running_total, 3),
            'val_rmse': round(self.val_running_rmse/self.val_running_total, 3),
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def on_train_epoch_end(self):
        self.running_true, self.running_total = 0, 0
        self.running_rmse = 0

    
    def on_train_epoch_start(self) -> None:
        print('\n')
        self.running_true, self.running_total = 0, 0
        self.running_rmse = 0
        print(f'running true: {self.running_true}, running total: {self.running_total}')

    
    def on_validation_epoch_end(self) -> None:
        self.val_running_true, self.val_running_total = 0, 0
        self.val_running_rmse = 0


    def on_validation_epoch_start(self) -> None:
        self.val_running_true, self.val_running_total = 0, 0
        self.val_running_rmse = 0

    


class EffUnet(BaseUnetModel):
    def __init__(self, general_cfg, model_cfg):
        super(EffUnet, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        encoder = torchvision.models.efficientnet_b0()

        self.conv_e1 = ConvBlock(self.config.in_c, 32, stride=2, act=nn.SiLU())    # /2, 32
        self.conv_e2 = encoder.features[1:3]     # /4, 24
        self.conv_e3 = encoder.features[3]       # /8, 40
        self.conv_e4 = encoder.features[4]       # /16, 80,    cua nos chac la [4:5] de co 112 channel
        self.conv_e5 = encoder.features[5:7]      # /32, 192
        self.conv_connect = nn.Sequential(
            ConvBlock(in_c=192, out_c=192, act=nn.SiLU()),
            ConvBlock(in_c=192, out_c=192, act=nn.SiLU()),
        )

        self.conv_d1 = nn.Sequential(
            ConvBlock(in_c=272, out_c=80, act=nn.SiLU()),
            ConvBlock(in_c=80, out_c=40, act=nn.SiLU())
        )
        self.conv_d2 = nn.Sequential(
            ConvBlock(in_c=80, out_c=80, act=nn.SiLU()),
            ConvBlock(in_c=80, out_c=24, act=nn.SiLU())
        )
        self.conv_d3 = nn.Sequential(
            ConvBlock(in_c=48, out_c=48, act=nn.SiLU()),
            ConvBlock(in_c=48, out_c=32, act=nn.SiLU())
        )
        self.conv_d4 = nn.Sequential(
            ConvBlock(in_c=64, out_c=64, act=nn.SiLU()),
            ConvBlock(in_c=64, out_c=32, act=nn.SiLU())
        )
        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
        #     nn.SiLU(),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1),
        #     nn.Sigmoid()
        # )

        c=32
        self.head = nn.Sequential(
            # ConvBlock(in_c=c, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c, act=nn.SiLU()),

            # nn.Conv2d(c, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(c*4, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),

            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, input):
        e1 = self.conv_e1(input)     # /2, 32
        e2 = self.conv_e2(e1)     # /4, 24
        e3 = self.conv_e3(e2)       # /8, 40
        e4 = self.conv_e4(e3)       # /16, 80
        e5 = self.conv_e5(e4)      # /32, 192
        e5 = self.conv_connect(e5)   # /32, 192

        d1 = torch.concat([e4, F.interpolate(e5, scale_factor=2, mode='bilinear')], dim=1)
        d1 = self.conv_d1(d1)

        d2 = torch.concat([e3, F.interpolate(d1, scale_factor=2, mode='bilinear')], dim=1)
        d2 = self.conv_d2(d2)

        d3 = torch.concat([e2, F.interpolate(d2, scale_factor=2, mode='bilinear')], dim=1)
        d3 = self.conv_d3(d3)

        d4 = torch.concat([e1, F.interpolate(d3, scale_factor=2, mode='bilinear')], dim=1)
        d4 = self.conv_d4(d4)

        d5 = F.interpolate(d4, scale_factor=2, mode='bilinear')
        out = self.head(d5)

        return out


from .unet_utils import *
class EffSmpUnet(BaseUnetModel):
    def __init__(self, general_cfg, model_cfg):
        super(EffSmpUnet, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        encoder = torchvision.models.efficientnet_b0()

        self.conv_e1 = ConvBlock(self.config.in_c, 32, stride=2, act=nn.SiLU())    # /2, 32
        self.conv_e2 = encoder.features[1:3]     # /4, 24
        self.conv_e3 = encoder.features[3]       # /8, 40
        self.conv_e4 = encoder.features[4:6]       # /16, 80,    cua nos chac la [4:5] de co 112 channel
        self.conv_e5 = encoder.features[6:8]      # /32, 192
        
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels = (15, 32, 24, 40, 112, 320),
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type='scse',
        )

        c = 16
        self.hm_out = nn.Sequential(
            # ConvBlock(in_c=c, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c*4, act=nn.SiLU()),
            # ConvBlock(in_c=c*4, out_c=c, act=nn.SiLU()),

            # nn.Conv2d(c, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(c*4, c*4, kernel_size=3, stride=1, padding=1),
            # nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),

            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, input):
        e0 = input
        e1 = self.conv_e1(input)     # /2, 32
        e2 = self.conv_e2(e1)     # /4, 24
        e3 = self.conv_e3(e2)       # /8, 40
        e4 = self.conv_e4(e3)       # /16, 80
        e5 = self.conv_e5(e4)      # /32, 192

        decoder_output = self.decoder(e0, e1, e2, e3, e4, e5)
        out = self.hm_out(decoder_output)

        return out
    


class SmpUnet(BaseUnetModel):
    def __init__(self, general_cfg, model_cfg):
        super(SmpUnet, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        self.model = smp.UnetPlusPlus(
            encoder_name=self.config.backbone, 
            encoder_depth=self.config.encoder_depth,
            encoder_weights="imagenet",
            in_channels=self.config.in_c,                  
            classes=1,
        )
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input):
        out = self.model(input)
        out = F.sigmoid(out)
        return out
    

class SmpUnetModified(BaseUnetModel):
    def __init__(self, general_cfg, model_cfg):
        super(SmpUnetModified, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        base_model = smp.UnetPlusPlus(
            encoder_name=self.config.backbone, 
            encoder_depth=self.config.encoder_depth,
            encoder_weights="imagenet",
            in_channels=self.config.in_c,                  
            classes=1,
        )

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder

        c = self.decoder.blocks.x_0_4.conv2[0].out_channels

        self.hm_out = nn.Sequential(
            ConvBlock(in_c=c, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c),
            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, input):
        features = self.encoder(input)
        decoder_output = self.decoder(*features)
        hm = self.hm_out(decoder_output)
        return hm

    
class SmpUnetEvent(BaseUnetModel):
    def __init__(self, general_cfg, model_cfg):
        super(SmpUnetEvent, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        base_model = smp.UnetPlusPlus(
            encoder_name=self.config.backbone, 
            encoder_depth=self.config.encoder_depth,
            encoder_weights="imagenet",
            in_channels=self.config.in_c,                  
            classes=1,
        )

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder

        c = self.decoder.blocks.x_0_4.conv2[0].out_channels

        self.hm_out = nn.Sequential(
            ConvBlock(in_c=c, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c),
            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.event_out = nn.Sequential(
            ConvBlock(in_c=c, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c*4),
            ConvBlock(in_c=c*4, out_c=c*4),

            ConvBlock(c*4, c*4),
            nn.Dropout(p=0.1),
            ConvBlock(c*4, c*4),
            nn.Dropout(p=0.1),
            ConvBlock(c*4, c*4),
            nn.Dropout(p=0.1),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=c*4, out_features=c*2),
            nn.SiLU(),
            nn.Linear(in_features=c*2, out_features=self.config.n_class),
        )
        

    def forward(self, input):
        features = self.encoder(input)
        decoder_output = self.decoder(*features)
        hm = self.hm_out(decoder_output)
        event = self.event_out(decoder_output)
        return hm, event
    

class SmpDeepLab(BaseUnetModel):
    def __init__(self, general_cfg, model_cfg):
        super(SmpDeepLab, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()
    

    def _init_layers(self):
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.config.backbone, 
            encoder_depth=self.config.encoder_depth,
            encoder_weights="imagenet",
            in_channels=self.config.in_c,                  
            classes=1,
        )
        

    def forward(self, input):
        out = self.model(input)
        out = torch.sigmoid(out)
        return out

    
if __name__ == '__main__':
    from config import *
    general_cfg.data.output_stride = 1
    model = EffSmpUnet(general_cfg, smpunet_cfg)
    x = torch.rand(1, 15, 512, 512)
    hm = model(x)
    pdb.set_trace()
    print(hm.shape)