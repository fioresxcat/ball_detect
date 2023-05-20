import torch
import torch.nn as nn
import yaml
from models.unet import *
from models.centernet import *
from dataset import *
from dataset_event import *
from my_utils import *
from config import general_cfg
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import shutil
import pdb


def train(general_cfg, model_cfg):
    if general_cfg.training.prev_ckpt_path is None:
        experiment_dir = get_experiment_dir(general_cfg.training.ckpt_save_dir, description=general_cfg.training.exp_description)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'general_cfg.json'), 'w') as f:
            json.dump(general_cfg, f, indent=4)

        with open(os.path.join(experiment_dir, 'model_cfg.json'), 'w') as f:
            json.dump(model_cfg, f, indent=4)
    else:
        experiment_dir = str(Path(general_cfg.training.prev_ckpt_path).parent)

    # get data
    if general_cfg.data.train_event:
        data_module = BallDataEventModule(general_cfg=general_cfg, augment=general_cfg.training.augment)
    else:
        data_module = BallDataModule(general_cfg=general_cfg, augment=general_cfg.training.augment)

    print('NUM TRAIN SAMPLES: ', len(data_module.train_ds))
    print('NUM VAL SAMPLES: ', len(data_module.val_ds))

    # init model
    model = load_model(
        model_type=model_cfg.name,
        general_cfg=general_cfg,
        model_cfg=model_cfg,
        ckpt_path=None
    )
    pdb.set_trace()

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        dirpath=experiment_dir,
        filename='model-{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-{val_acc:.3f}-{val_rmse:.3f}',
        save_top_k=3,
        auto_insert_metric_name=True,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop = EarlyStopping(
        monitor="val_acc", 
        mode="max",
        stopping_threshold=1,
        patience=10,
    )
    rich_prog = RichProgressBar(leave=True)
    rich_summary = RichModelSummary()

    # tensorboard logger
    logger = TensorBoardLogger(
        save_dir=experiment_dir,
        name='',
        version=''
    )
    
    # trainer
    trainer = Trainer(
        accelerator='gpu',
        gpus='0',
        max_epochs=general_cfg.training.max_epoch,
        min_epochs=general_cfg.training.min_epoch,
        auto_scale_batch_size=True,
        callbacks=[model_ckpt, lr_monitor, early_stop, rich_prog, rich_summary],
        logger=logger,
        log_every_n_steps=50,
        precision=general_cfg.training.precision,
        profiler='simple'
        # fast_dev_run=True,
        # overfit_batches=1,
    )

    # train
    if general_cfg.training.prev_ckpt_path is not None:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=general_cfg.training.prev_ckpt_path)
    else:
        trainer.fit(model=model, datamodule=data_module)

    print(f'Training done.')


if __name__ == '__main__':
    from config import *

    train(general_cfg, centernet_yolo_cfg)

