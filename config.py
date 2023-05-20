from easydict import EasyDict


general_cfg = EasyDict({
    'data': {
        'train_dir': '/data/tungtx2/datn/ttnet/dataset/train',
        'test_dir':'/data/tungtx2/datn/ttnet/dataset/test',
        'n_sample_limit': 1e9,
        'orig_size': (1920, 1080),
        'input_size': (512, 512),
        'ball_radius': (7, 7),
        'mask_radius': (10, 10),
        'n_input_frames': 5,
        'train_event': False,
        'output_stride': 4,
    },
    
    'training': {
        'exp_description': 'centernet_yolov8n_test',
        'mask_ball_prob': 0.15,
        'prev_ckpt_path': None,
        'base_lr': 0.001,
        'bs': 8,
        'max_epoch': 100,
        'min_epoch': 30,
        'precision': 16,


        'shuffle_train': True,
        'augment': True,
        'augment_prob': 0.5,
        'ckpt_save_dir': 'ckpt',
        'weight_decay': 1e-2,
        'use_warmup': False,
        'warmup_ratio': 0.1,
        'num_workers': 8
    },

    'decode': {
        'kernel': 3,
        'conf_thresh': 0.5,
        'rmse_thresh': 3,   # on 512 x 512 input
        'decode_by_area': False,
        'ev_acc_thresh': 0.25
    },


    'centernet_pytorch': {
        'slug': 'rx50',
        'fpn': False,
        'bn_momentum': 0.1,
        'head_channel': 64,
        'num_classes': 1,
    },
    
})

smpunet_cfg = EasyDict({
    'name': 'smpunet',
    # 'backbone': 'efficientnet-b0',
    'backbone': 'resnet34',
    'encoder_depth': 5,
    'loss': 'modified_focal_loss',
})

smpunet_modified_cfg = EasyDict({
    'name': 'smpunet_modified',
    'backbone': 'efficientnet-b0',
    'encoder_depth': 5,
    'loss': 'modified_focal_loss',
})


smpunet_event_cfg = EasyDict({
    'name': 'smpunet',
    'backbone': 'efficientnet-b0',
    'encoder_depth': 5,
    'loss': 'modified_focal_loss',
    'n_class': 2,
    'freeze_event': True,
    'freeze_ball': False,
    'l_ball': 1,
    'l_event': 1,
    'bounce_weight': 1,
    'net_weight': 2
})

smpdeeplab_cfg = EasyDict({
    'name': 'smpdeeplab',
    'backbone': 'efficientnet-b0',
    'encoder_depth': 5,
    'loss': 'modified_focal_loss',
})


centernet_yolo_cfg = EasyDict({
    'name': 'centernet_yolo',
    'version': 'n',
    'loss': 'centernet_loss',
    'load_pretrained_yolov8': True,
})


centernet_yolo_event_cfg = EasyDict({
    'name': 'centernet_yolo_event',
    'version': 'n',
    'loss': 'centernet_event_loss',
    'load_pretrained_yolov8': True,
    'freeze_event': True,
    'freeze_ball': False,
    'l_ball': 1,
    'l_event': 1,
    'bounce_weight': 1,
    'net_weight': 2
})

centernet_hourglass_cfg = EasyDict({
    'name': 'centernet_hourglass',
    'loss': 'centernet_loss',
})

# set data_dict_path
if not general_cfg.data.train_event:
    general_cfg.data.train_dict_path = f'data/gpu2_train_dict_{general_cfg.data.n_input_frames}.pkl'
    general_cfg.data.val_dict_path = f'data/gpu2_val_dict_{general_cfg.data.n_input_frames}.pkl'
    general_cfg.data.test_dict_path = f'data/gpu2_test_dict_{general_cfg.data.n_input_frames}.pkl'
else:
    general_cfg.data.train_dict_path = f'data/gpu2_event_train_dict_{general_cfg.data.n_input_frames}.pkl'
    general_cfg.data.val_dict_path = f'data/gpu2_event_val_dict_{general_cfg.data.n_input_frames}.pkl'
    general_cfg.data.test_dict_path = f'data/gpu2_event_test_dict_{general_cfg.data.n_input_frames}.pkl'

# set in_c
smpunet_cfg.in_c = general_cfg.data.n_input_frames * 3
smpunet_modified_cfg.in_c = general_cfg.data.n_input_frames * 3
smpunet_event_cfg.in_c = general_cfg.data.n_input_frames * 3
smpdeeplab_cfg.in_c = general_cfg.data.n_input_frames * 3
centernet_yolo_cfg.in_c = general_cfg.data.n_input_frames * 3
centernet_yolo_event_cfg.in_c = general_cfg.data.n_input_frames * 3
centernet_hourglass_cfg.in_c = general_cfg.data.n_input_frames * 3

# set output stride
smpunet_cfg.output_stride = general_cfg.data.output_stride
smpdeeplab_cfg.output_stride = general_cfg.data.output_stride
centernet_yolo_cfg.output_stride = general_cfg.data.output_stride
centernet_yolo_event_cfg.output_stride = general_cfg.data.output_stride
centernet_hourglass_cfg.output_stride = general_cfg.data.output_stride



if __name__ == '__main__':
    import pdb

    print(general_cfg.data.test_dict_path)
    pdb.set_trace()