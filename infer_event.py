import pdb
import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_utils import load_model
from dataset_event import *
from config import *
from tqdm import tqdm


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model_type = 'centernet_yolo_event'
    ckpt_path = 'ckpt/exp55_centernet_only_bounce/model-epoch=24-train_loss=0.174-val_loss=0.164-val_acc=0.994-val_ev_acc=0.999-val_ev_loss=0.115-val_rmse=5.539.ckpt'
    model_cfg = centernet_yolo_event_cfg

    model = load_model(
        model_type,
        general_cfg,
        model_cfg,
        ckpt_path
    )
    model.eval().to(device)

    bs = 16
    ds = BallDatasetEvent(general_cfg, transforms=None, mode='test', augment=False)
    ds_loader = DataLoader(ds, shuffle=False, batch_size=bs)

    true, total = 0, 0
    for item_idx, item in enumerate(tqdm(ds_loader)):
        batch_img_paths = ds.ls_img_paths[item_idx*bs:(item_idx+1)*bs]
        imgs, hm, om, out_pos, norm_pos, ev_true = item
        imgs = imgs.to(device)
        with torch.no_grad():
            hm_pred, om_pred, ev_pred = model(imgs)
        ev_pred = torch.sigmoid(ev_pred)

        diff = torch.abs(ev_pred[:, 0] - ev_true[:, 0].to(device))   # shape nx2, event_pred is not sigmoided inside the model
        n_true = (diff < 0.25).sum()
        true += n_true.cpu().item()
        total += ev_pred.shape[0]

        print(f'true: {true}, total: {total}, acc: {round(true/total, 3)}')
        # for p, t in list(zip(ev_pred, ev_true)):
        #     print(torch.round(p, decimals=3), torch.round(t, decimals=3))
        # # break

    