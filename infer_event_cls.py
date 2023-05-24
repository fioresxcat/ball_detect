import pdb
import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_utils import load_model
from tqdm import tqdm
from config_event_cls import *
from dataset_event_cls import *


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model_type = 'my_event_cls'
    ckpt_path = 'ckpt_event_cls/exp3_centernet_v8n/model-epoch=20-train_loss=0.495-val_loss=0.549-val_ev_acc=0.458.ckpt'
    model_cfg = my_event_cls_cfg

    model = load_model(
        model_type,
        general_ev_cfg,
        model_cfg,
        ckpt_path
    )
    model.eval().to(device)

    bs = 1
    ds = EventClassifyDataset(general_ev_cfg, mode='train')
    ds_loader = DataLoader(ds, shuffle=False, batch_size=bs)

    for item_idx, item in enumerate(tqdm(ds_loader)):
        x, ev_true = item
        if ev_true[0][0].item() == 0 and ev_true[0][1] == 1 :
            x = x.to(device).to(torch.float)
            with torch.no_grad():
                ev_pred = model(x)
            ev_pred = torch.sigmoid(ev_pred)
            for p, t in list(zip(ev_pred, ev_true)):
                print(torch.round(p.to(torch.float), decimals=3), torch.round(t.to(torch.float), decimals=3))
        # break

    