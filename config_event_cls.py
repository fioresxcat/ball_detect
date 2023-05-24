from easydict import EasyDict

general_ev_cfg = EasyDict({
    'data': {
        'root_dir': '/data2/tungtx2/datn/ball_detect/data_event_cls/centernet_v8n/',
        'train_dir': '/data2/tungtx2/datn/ball_detect/data_event_cls/centernet_v8n/train',
        'val_dir':'/data2/tungtx2/datn/ball_detect/data_event_cls/centernet_v8n/val',
        'test_dir':'/data2/tungtx2/datn/ball_detect/data_event_cls/centernet_v8n/test',
        'n_sample_limit': 1e9,
    },
    
    'training': {
        'exp_description': 'centernet_v8n',
        'prev_ckpt_path': None,
        'base_lr': 0.001,
        'bs': 256,
        'max_epoch': 100,
        'min_epoch': 30,
        'precision': 16,
        'monitor': 'val_ev_acc',
        'monitor_mode': 'max',

        'shuffle_train': True,
        'augment': True,
        'augment_prob': 0.5,
        'ckpt_save_dir': 'ckpt_event_cls',
        'weight_decay': 1e-2,
        'use_warmup': False,
        'warmup_ratio': 0.1,
        'num_workers': 8
    },

    'decode': {
        'ev_diff_thresh': 0.25
    },
})

my_event_cls_cfg = EasyDict({
    'name': 'my_event_cls',
    'in_c': 5,
    'n_class': 2,
    'l_ball': 1,
    'l_event': 1,
    'bounce_weight': 1,
    'net_weight': 2
})