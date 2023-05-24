import torch
from turbojpeg import TurboJPEG
from config import *
from dataset_event import BallDatasetEvent
from torch.utils.data import DataLoader
from my_utils import *
from tqdm import tqdm
from collections import deque


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model_type = 'centernet_yolo'
    ckpt_path = 'ckpt/exp38_centernet_ason_v8n_fixed_mask_ball/model-epoch=18-train_loss=0.190-val_loss=0.160-val_acc=0.994-val_rmse=0.361.ckpt'
    model_cfg = centernet_yolo_cfg
    model_cfg.in_c = 15
    
    save_dir = 'data_event_cls'
    bs = 8
    mode = 'test'
    description = 'centernet_v8n'

    ds = BallDatasetEvent(general_cfg=general_cfg, transforms=None, mode=mode, augment=False)
    data_loader = DataLoader(ds, batch_size=bs, shuffle=False)
    model = load_model(
        model_type,
        general_cfg,
        model_cfg,
        ckpt_path
    )
    model.eval().to(device)

    os.makedirs(os.path.join(save_dir, description, mode), exist_ok=True)

    cnt = 0
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):       # img_paths have 9 img_fp
        batch_img_paths = ds.ls_img_paths[batch_idx*bs:(batch_idx+1)*bs]
        batch_imgs, _, _, _, _, batch_ev_true = batch

        ls_imgs = []
        for img_9s in batch_imgs:
            idx = 0
            while idx <= 12:
                img_5s = img_9s[0+idx:15+idx]
                ls_imgs.append(img_5s)
                idx += 3
        ls_imgs = torch.stack(ls_imgs, dim=0).to(device)
        # print('ls_imgs shape: ', ls_imgs.shape)
        with torch.no_grad():
            batch_pred, _ = model(ls_imgs)  # shape 1 x 1 x 512 x 512

        batch_pred = batch_pred.squeeze(1).cpu().numpy()  # (bs*5)  x 512 x 512
        batch_pred = np.split(batch_pred, bs, axis=0)   # [(5  x 512 x 512)] * bs
        batch_ev_true = batch_ev_true.cpu().numpy()     # shape bs x 2

        for i, pred in enumerate(batch_pred):
            ev_true = batch_ev_true[i]
            # print('pred shape: ', pred.shape)
            # print('label shape: ', ev_true.shape)

            img_paths = batch_img_paths[i]
            game_name = img_paths[0].split('/')[-2]
            start_name = game_name + '_' + Path(img_paths[0]).stem
            end_name = Path(img_paths[-1]).stem
            np.savez(os.path.join(save_dir, description, mode, f'{start_name}-{end_name}.npz'), input=pred.astype(np.float16), label=ev_true.astype(np.float16))
            # print('done')
