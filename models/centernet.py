import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from .model_utils import *
from .basemodel import BaseModel
from loss import CenterNetLoss, CenterNetEventLoss, CenterNetLossMultiBall
from centernet_yolo.model import CenterNetYolov8, CenterNetYolov8Event
import torchmetrics
import torchvision


criterion_dict = {
    'centernet_loss': CenterNetLoss,
    'centernet_loss_multi_ball': CenterNetLossMultiBall,
    'centernet_event_loss': CenterNetEventLoss,
}

class MyAccuracy(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        max_preds, max_pred_indices = torch.max(preds, dim=1)
        valid_pred_indices = max_pred_indices[max_preds>=0.5]
        max_target, max_target_indices = torch.max(target, dim=1)
        valid_target_indices = max_target_indices[max_preds>=0.5]

        # n_true = (valid_pred_indices==valid_target_indices).sum()
        n_true = (max_pred_indices==max_target_indices).sum()

        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total
    


class BaseCenterNetModel(BaseModel):
    def __init__(self, general_cfg, model_cfg):
        super().__init__(general_cfg)
        self.config = model_cfg
        self._init_lightning_stuff()
    

    def _init_lightning_stuff(self):
        self.criterion = criterion_dict[self.config.loss](
            pos_pred_weight = self.config.pos_pred_weight,
        )
        self._reset_metrics()

    def _reset_metrics(self):
        self.running_true, self.running_total, self.running_rmse = 0, 0, 0
        self.val_running_true, self.val_running_total, self.val_running_rmse = 0, 0, 0
        self.test_running_true, self.test_running_total, self.test_running_rmse = 0, 0, 0
    

    def common_step(self, batch, batch_idx):
        imgs, hm_true, om_true, pos, normalized_pos = batch
        hm_pred, om_pred = self.forward(imgs)
        loss = self.criterion(hm_pred, om_pred, hm_true, om_true, pos)
        return hm_pred, hm_true, loss
        
    

    def training_step(self, batch, batch_idx):
        hm_pred, hm_true, loss = self.common_step(batch, batch_idx)

        # compute metrics
        if not self.general_cfg.training.multi_ball:
            n_true, sum_batch_rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, rmse_thresh=2)
            self.running_true += n_true
            self.running_total += hm_pred.shape[0]
            self.running_rmse += sum_batch_rmse

            self.log_dict({
                'train_loss': loss,
                'train_acc': torch.round(self.running_true/self.running_total, decimals=3),
                'train_rmse': torch.round(self.running_rmse/self.running_total, decimals=3),
            }, on_step=True, on_epoch=True, prog_bar=True)
        
        else:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        hm_pred, hm_true, loss = self.common_step(batch, batch_idx)

        n_true, sum_batch_rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, rmse_thresh=2)
        self.val_running_true += n_true
        self.val_running_total += hm_pred.shape[0]
        self.val_running_rmse += sum_batch_rmse

        self.log_dict({
            'val_loss': loss,
            'val_acc': torch.round(self.val_running_true/self.val_running_total, decimals=3),
            'val_rmse': torch.round(self.val_running_rmse/self.val_running_total, decimals=3),
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def test_step(self, batch, batch_idx):
        hm_pred, hm_true, loss = self.common_step(batch, batch_idx)

        n_true, sum_batch_rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, rmse_thresh=2)
        self.test_running_true += n_true
        self.test_running_total += hm_pred.shape[0]
        self.test_running_rmse += sum_batch_rmse

        self.log_dict({
            'test_loss': loss,
            'test_acc': torch.round(self.test_running_true/self.test_running_total, decimals=3),
            'test_rmse': torch.round(self.test_running_rmse/self.test_running_total, decimals=3),
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    


    def on_train_epoch_end(self):
        self._reset_metrics()
    
    def on_train_epoch_start(self) -> None:
        print('\n')
        self._reset_metrics()
    
    def on_validation_epoch_end(self) -> None:
        self._reset_metrics()

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def on_train_start(self) -> None:
        if self.config.reset_optimizer:
            default_cfg = self.trainer.optimizers[0].defaults
            default_cfg.pop('differentiable')
            opt = type(self.trainer.optimizers[0])(self.parameters(), **default_cfg)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reseted')



class BaseCenterNetMultiBallModel(BaseCenterNetModel):
    def __init__(self, general_cfg, model_cfg):
        super().__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_lightning_stuff()
    


    def common_step(self, batch, batch_idx):
        imgs, hm_true, om_true = batch
        hm_pred, om_pred = self.forward(imgs)
        loss = self.criterion(hm_pred, om_pred, hm_true, om_true)
        return hm_pred, hm_true, loss
        


class BaseCenterNetEventModel(BaseModel):
    def __init__(self, general_cfg, model_cfg):
        super(BaseCenterNetEventModel, self).__init__(general_cfg)
        self.config = model_cfg
        self._init_lightning_stuff()
    

    def _init_lightning_stuff(self):
        self.criterion = criterion_dict[self.config.loss](
            only_bounce = self.general_cfg.data.only_bounce,
            l_event = 0 if self.config.freeze_event else self.config.l_event,
            l_ball = 0 if self.config.freeze_ball else self.config.l_ball,
            bounce_pos_weight = self.config.bounce_pos_weight,
            bounce_weight = self.config.bounce_weight,
            net_weight = self.config.net_weight,
            empty_weight = self.config.empty_weight
        )
        self.train_ev_acc = MyAccuracy()
        self.val_ev_acc = MyAccuracy()
        self.test_ev_acc = MyAccuracy()

        self._reset_manual_metric()


    def configure_optimizers(self):
        base_lr = self.general_cfg.training.base_lr
        opt = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=self.general_cfg.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.general_cfg.training.monitor_mode,
            factor=0.2,
            patience=7,
            verbose=True
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     opt, 
        #     step_size=7,
        #     gamma=0.3
        # )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.general_cfg.training.monitor,
                'frequency': 1,
                'interval': 'epoch'
            }
        }
    

    def _reset_manual_metric(self):
        self.running_ball_true, self.running_ball_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        self.val_running_ball_true, self.val_running_ball_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)

        # self.running_ev_true, self.running_ev_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        # self.val_running_ev_true, self.val_running_ev_total = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        
        self.running_rmse = torch.tensor(0, device=self.device)
        self.val_running_rmse = torch.tensor(0, device=self.device)


    def common_step(self, batch, batch_idx):
        imgs, hm_true, om_true, out_pos, normalized_pos, event_true = batch
        hm_pred, om_pred, event_pred = self.forward(imgs)
        loss, ball_loss, ev_loss = self.criterion(hm_pred, om_pred, event_pred, hm_true, om_true, event_true, out_pos)
        return hm_pred, om_pred, event_pred, loss, ball_loss, ev_loss
    

    def training_step(self, batch, batch_idx):
        imgs, hm_true, om_true, out_pos, normalized_pos, event_true = batch
        hm_pred, om_pred, event_pred, loss, ball_loss, ev_loss = self.common_step(batch, batch_idx)

        # compute ball detect metrics
        n_true, rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, rmse_thresh=2)
        self.running_ball_true += n_true
        self.running_ball_total += hm_pred.shape[0]

        # compute event metric
        if not self.config.freeze_event:
            self.train_ev_acc(event_pred, event_true)

        self.log_dict({
            'train_loss': loss,
            'train_acc': torch.round(self.running_ball_true/self.running_ball_total, decimals=3),
            'train_rmse': rmse,
            'train_ev_loss': ev_loss,
            'train_ev_acc': self.train_ev_acc if not self.config.freeze_event else -1,
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        imgs, hm_true, om_true, out_pos, normalized_pos, event_true = batch
        hm_pred, om_pred, event_pred, loss, ball_loss, ev_loss = self.common_step(batch, batch_idx)

        # compute ball detect metrics
        n_true, rmse = compute_metrics(hm_pred, hm_true, self.general_cfg.decode.kernel, self.general_cfg.decode.conf_thresh, rmse_thresh=2)
        self.val_running_ball_true += n_true
        self.val_running_ball_total += hm_pred.shape[0]

        # compute event metric
        if not self.config.freeze_event:
            self.val_ev_acc(event_pred, event_true)

        self.log_dict({
            'val_loss': loss,
            'val_acc': torch.round(self.val_running_ball_true/self.val_running_ball_total, decimals=3),
            'val_rmse': rmse,
            'val_ev_loss': ev_loss,
            'val_ev_acc': self.val_ev_acc if not self.config.freeze_event else -1,
        }, on_step=True, on_epoch=True, prog_bar=True)



        return loss
    

    
    def on_train_start(self) -> None:
        print('on train start...')
        if self.config.reset_optimizer:
            initial_opt_params = self.trainer.optimizers[0].defaults
            initial_opt_params.pop('differentiable')
            opt = type(self.trainer.optimizers[0])(self.parameters(), **initial_opt_params)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reseted')




class CenterNetHourGlass(BaseCenterNetModel):
    def __init__(self, general_cfg, model_cfg):
        super(CenterNetHourGlass, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()


    def _init_layers(self):
        self.conv1 = ConvBlock(in_c=self.config.in_c, out_c=128, stride=2, act=nn.ReLU())
        self.res1 = ResBlock(in_c=128, out_c=96, stride=2)
        self.hour_glass_104 = nn.Sequential(
            HourGlassModule(in_c=96, out_c=96),
            HourGlassModule(in_c=96, out_c=96),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, input):
        x = self.conv1(input)
        x = self.res1(x)
        x = self.hour_glass_104(x)
        heatmap_out, offset_out = self.heatmap_head(x), self.offset_head(x)
        return heatmap_out, offset_out

class CenterNetYolo(BaseCenterNetModel):
    def __init__(self, general_cfg, model_cfg):
        super(CenterNetYolo, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()


    def _init_layers(self):
        self.model = CenterNetYolov8(
            version=self.config.version,
            nc=1,
            load_pretrained_yolov8=self.config.load_pretrained_yolov8
        )
        self.model.backbone.backbone[0].conv = nn.Conv2d(self.config.in_c, self.model.backbone.backbone[0].conv.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


    def forward(self, input):
        hm_pred, om_pred = self.model(input)
        return hm_pred, om_pred
    
from test_yolov8_p2 import CenterNetYolov8_P2_Flow, CenterNetYolov8_P2
class CenterNetYolo_P2_Flow(BaseCenterNetModel):
    def __init__(self, general_cfg, model_cfg):
        super(CenterNetYolo_P2_Flow, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()


    def _init_layers(self):
        self.model = CenterNetYolov8_P2_Flow(
            version=self.config.version,
            nc=1,
            load_pretrained_yolov8=self.config.load_pretrained_yolov8
        )
        self.model.backbone.backbone[0].conv = nn.Conv2d(self.config.in_c, self.model.backbone.backbone[0].conv.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


    def forward(self, input):
        hm_pred, om_pred = self.model(input)
        return hm_pred, om_pred


class CenterNetYolo_P2(BaseCenterNetModel):
    def __init__(self, general_cfg, model_cfg):
        super(CenterNetYolo_P2, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()


    def _init_layers(self):
        self.model = CenterNetYolov8_P2(
            version=self.config.version,
            nc=1,
            load_pretrained_yolov8=self.config.load_pretrained_yolov8
        )
        self.model.backbone.backbone[0].conv = nn.Conv2d(self.config.in_c, self.model.backbone.backbone[0].conv.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


    def forward(self, input):
        hm_pred, om_pred = self.model(input)
        return hm_pred, om_pred
    


class CenterNetYoloMultiBall(BaseCenterNetMultiBallModel):
    def __init__(self, general_cfg, model_cfg):
        super(CenterNetYoloMultiBall, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()


    def _init_layers(self):
        self.model = CenterNetYolov8(
            version=self.config.version,
            nc=1,
            load_pretrained_yolov8=self.config.load_pretrained_yolov8
        )
        self.model.backbone.backbone[0].conv = nn.Conv2d(self.config.in_c, self.model.backbone.backbone[0].conv.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


    def forward(self, input):
        hm_pred, om_pred = self.model(input)
        return hm_pred, om_pred
    


class CenterNetYoloEvent(BaseCenterNetEventModel):
    def __init__(self, general_cfg, model_cfg):
        super(CenterNetYoloEvent, self).__init__(general_cfg, model_cfg)
        self.config = model_cfg
        self._init_layers()


    def _init_layers(self):
        self.model = CenterNetYolov8Event(
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



    def forward(self, input):
        hm_pred, om_pred, ev_pred = self.model(input)
        return hm_pred, om_pred, ev_pred

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


    def forward(self, input):
        return self.model(input)
    

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
            mode=self.general_cfg.training.monitor_mode,
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


    
if __name__ =='__main__':
    from config import *
    from my_utils import load_model, load_state_dict_for_only_bounce_model

    model_type = 'centernet_yolo_event'
    ckpt_path = '/data2/tungtx2/datn/ball_detect/ckpt/exp52_centernet_event_v8n_finetune_all_from_exp_40/model-epoch=106-train_loss=0.574-val_loss=0.572-val_acc=0.997-val_ev_acc=0.507-val_rmse=2.405.ckpt'
    model_cfg = centernet_yolo_event_cfg

    trained_model = load_model(
        model_type,
        general_cfg,
        model_cfg,
        ckpt_path
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    pdb.set_trace()

    model = CenterNetYoloEventOnlyBounce(general_cfg, centernet_yolo_event_only_bounce_cfg)
    model = load_state_dict_for_only_bounce_model(
        model,
        ckpt_dir='/data2/tungtx2/datn/ball_detect/ckpt/exp_52_ep_106'
    )

    pdb.set_trace()
    x = torch.rand(1, 27, 512, 512)
    hm, om, ev = model(x)
    print(hm.shape, om.shape, ev.shape)