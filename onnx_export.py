import onnx, onnxruntime
import torch
import numpy as np
from config import *
from my_utils import *
import pdb

if __name__ == '__main__':
    device = 'cpu'
    model_type = 'centernet_yolo_no_asl'
    ckpt_path = '/data2/tungtx2/datn/ball_detect/ckpt/exp85_centernet_no_asl_3_frames_add_pos_pred_weight_add_no_ball_frame/model-epoch=43-train_loss=0.136-val_loss=0.106-val_acc=0.998-val_ev_acc=0.000-val_ev_loss=0.000-val_rmse=0.194.ckpt'
    model_cfg = centernet_yolo_no_asl_cfg
    model_cfg.load_pretrained_yolov8 = False

    general_cfg.data.n_input_frames = 3
    general_cfg.data.train_event = False

    model = load_model(
        model_type,
        general_cfg,
        model_cfg,
        ckpt_path
    )
    model.eval().to(device)
    x = torch.rand(1, 3*general_cfg.data.n_input_frames, 512, 512).to(device)
    with torch.cuda.amp.autocast(enabled=True):
        torch.onnx.export(
            model,
            x,
            'ckpt/exp85_centernet_no_asl_3_frames_add_pos_pred_weight_add_no_ball_frame/exp85_ep43_centernet_no_asl_3_frames.onnx',
            input_names=['imgs'],
            output_names=['hm', 'om'],
            do_constant_folding=True,
            dynamic_axes= {
                'imgs': {0: 'batch_size'},
                'hm': {0: 'batch_size'},
                'om': {0: 'batch_size'}
            },
            opset_version=14
        )