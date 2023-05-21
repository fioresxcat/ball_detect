import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
sys.path.append(os.getcwd())

import pdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import shutil
import time
from pathlib import Path
from my_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics import F1Score, Precision, Recall
from models.unet import *
from models.centernet import *
from config import *
from dataset import *
from loss import *
from my_utils import *
import math
from tqdm import tqdm


def l2_distance(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0]) ** 2 + (pt1[1]-pt2[1]) ** 2)



def get_peak_activation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Extracts the values from tensor A that are indexed by tensor B, where A has shape (N, C, H, W) and B has shape (N, 2).
    Returns a tensor of shape (N, C) containing the extracted values.
    """
    pdb.set_trace()
    N, H, W = A.shape

    # Compute the flat indices into tensor A corresponding to the indices in tensor B
    indices = B[:, 0] * W + B[:, 1]
    indices = indices.to(torch.int64)

    # Use torch.gather to extract the values in A indexed by B
    output = torch.gather(A.view(N, -1), dim=1, index=indices.unsqueeze(1))

    # Reshape the output tensor to the desired shape
    # output = output.view(N, C)

    return output


from models.model_utils import decode_hm, decode_hm_by_contour
def infer_unet(model, imgs, kernel, conf_thresh, decode_by_area=False):
    with torch.no_grad():
        hm = model(imgs)
    out_h, out_w = hm.shape[2:]
    in_h, in_w = imgs.shape[2:]
    assert in_h == out_h, f'input and output shape of unet does not match. in_h: {in_h}, out_h: {out_h}'
    assert in_w == out_w, f'input and output shape of unet does not match. in_w: {in_w}, out_w: {out_w}'

    if decode_by_area:
        batch_pos = decode_hm_by_contour(hm, conf_thresh)
    else:
        batch_pos = decode_hm(hm, kernel, conf_thresh)
    
    max_values, max_indices = torch.max(hm.squeeze(1).view(hm.shape[0], -1), dim=1)
    max_values = max_values.view(-1, 1)

    return batch_pos, max_values


def infer_centernet(model, imgs, kernel, conf_thresh, decode_by_area=False):
    with torch.no_grad():
        hm, om = model(imgs)
    out_h, out_w = hm.shape[2:]
    in_h, in_w = imgs.shape[2:]
    stride = in_h / out_h

    if decode_by_area:
        batch_pos = decode_hm_by_contour(hm, conf_thresh)
    else:
        batch_pos = decode_hm(hm, kernel, conf_thresh)    # shape nx2

    max_values, max_indices = torch.max(hm.squeeze(1).view(hm.shape[0], -1), dim=1)
    max_values = max_values.view(-1, 1)

    batch_final_pos = []
    for i in range(batch_pos.shape[0]):
        pos = batch_pos[i]
        offset = om[i][:, pos[1].long(), pos[0].long()]
        final_pos = pos + offset
        final_pos = final_pos.detach().cpu() / torch.tensor([out_w, out_h]) * torch.tensor([in_w, in_h])
        batch_final_pos.append(final_pos)
    batch_final_pos = torch.stack(batch_final_pos, dim=0).int()

    return batch_final_pos, max_values



if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model_type = 'effsmpunet'
    ckpt_path = 'ckpt/exp46_effsmpunet_silu_all_head_nho/model-epoch=12-train_loss=0.033-val_loss=0.039-val_acc=0.997-val_ev_acc=0.000-val_rmse=1.483.ckpt'
    model_cfg = effsmpunet_cfg
    save_dir = 'results/exp46_effsmpunet_silu_all_head_nho/test'
    draw_result = True
    bs = 32
    ds = BallDataset(general_cfg=general_cfg, transforms=None, mode='test')

    model = load_model(
        model_type,
        general_cfg,
        model_cfg,
        ckpt_path
    )
    model.eval().to(device)

    ds_loader = DataLoader(ds, batch_size=bs, shuffle=False)
    
    os.makedirs(save_dir, exist_ok=True)

    sum_rmse, n_true, n_total = 0, 0, 0
    res = EasyDict({
        'img_dict': EasyDict({})
    })

    ls_activation = []
    for item_idx, item in enumerate(tqdm(ds_loader)):
        batch_imgs, batch_hm_true, batch_om_true, batch_pos, batch_normalized_pos = item
        n, c, input_h, input_w = batch_imgs.size()
        batch_pos_true = (batch_normalized_pos * torch.tensor([input_w, input_h])).int()  # abs x, y in cv2 coord with img size 512
        batch_imgs = batch_imgs.to(device)


        if 'smp' in model_type:
            batch_pos_pred, batch_activation = infer_unet(model, batch_imgs, general_cfg.decode.kernel, general_cfg.decode.conf_thresh, general_cfg.decode.decode_by_area)      # pos_pred is abs coord with img size 512. Tensor. shape nx2
        elif 'centernet' in model_type:
            batch_pos_pred, batch_activation = infer_centernet(model, batch_imgs, general_cfg.decode.kernel, general_cfg.decode.conf_thresh, general_cfg.decode.decode_by_area)    # pos_pred is abs coord with img size 512. Tensor. shape nx2
        
        # pdb.set_trace()
        ls_activation.extend(batch_activation.cpu().numpy().tolist())
        batch_pos_true = batch_pos_true.to(device)
        batch_pos_pred = batch_pos_pred.to(device)
        batch_rmse = torch.sqrt(torch.pow(batch_pos_pred-batch_pos_true, 2).sum(dim=1))
        batch_res = (batch_rmse < general_cfg.decode.rmse_thresh).int()   # shape nx2, only 0,1. 0 means false and 1 means true

        sum_rmse += batch_rmse.sum().item()
        n_total += n
        n_true += batch_res.sum().item()

        batch_img_paths = ds.ls_img_paths[item_idx*bs:(item_idx+1)*bs]
        ls_img_path = [img_paths[-1] for img_paths in batch_img_paths]
        if draw_result:
            os.makedirs(os.path.join(save_dir, 'frame'), exist_ok=True)
            batch_imgs = (batch_imgs.permute(0, 2, 3, 1) * 255.).int().detach().cpu().numpy()
            for i, imgs in enumerate(batch_imgs):
                ls_img = np.split(imgs, general_cfg.data.n_input_frames, axis=2)   # list of n_input_frames imgs
                img = ls_img[-1].astype(np.uint8)    # last img which the prediction is for
                pos = batch_pos_pred[i]
                # pdb.set_trace()
                img_u = cv2.circle(cv2.UMat(img), (pos[0].int().item(), pos[1].int().item()), radius=5, color=(255, 0, 0), thickness=1)

                img_fp = Path(ls_img_path[i])   # get orig img path
                out_fp = os.path.join(save_dir, 'frame', f'{img_fp.parent.name}_{img_fp.name}')
                cv2.imwrite(out_fp, cv2.cvtColor(img_u.get(), cv2.COLOR_RGB2BGR))

        print('number of images: ', len(ds))
        print('true: ', n_true)
        print('total: ', n_total)
        print('acc: ', round(n_true/n_total, 3))
        print('mean rmse: ', round(sum_rmse/n_total, 3))

        for img_idx, img_fp in enumerate(ls_img_path):
            res.img_dict[img_fp] = EasyDict({
                'result': batch_res[img_idx].item(),
                'pred': (batch_pos_pred[img_idx][0].item(), batch_pos_pred[img_idx][1].item()),
                'true': (batch_pos_true[img_idx][0].item(), batch_pos_true[img_idx][1].item()),
            })

        # if item_idx == 10:
        #     break

    ls_activation = [el[0] for el in ls_activation]
    # save result
    res.num_images = len(ds)
    res.total = n_total
    res.true = n_true
    res.acc = round(n_true/n_total, 3)
    res.rmse_per_image = round(sum_rmse/n_total, 3)
    res.input_h = input_h
    res.input_w = input_w
    res.conf_thresh = general_cfg.decode.conf_thresh
    res.rmse_thresh = general_cfg.decode.rmse_thresh
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(res, f, indent=4)