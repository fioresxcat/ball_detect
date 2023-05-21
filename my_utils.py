import os
from pathlib import Path
import cv2
import numpy as np
import json
import pickle
import pdb
import albumentations as A
from models.unet import *
from models.centernet import *


SUPPORTED_MODEL = {
    'effunet': EffUnet,
    'effsmpunet': EffSmpUnet,
    'smpunet': SmpUnet,
    'smpunet_modified': SmpUnetModified,
    'smpdeeplab': SmpDeepLab,
    'centernet': CenterNetHourGlass,
    'centernet_yolo': CenterNetYolo,
    'centernet_yolo_event': CenterNetYoloEvent
}

def load_model(model_type, general_cfg, model_cfg, ckpt_path):
    if ckpt_path is not None:
        model = SUPPORTED_MODEL[model_type].load_from_checkpoint(
            ckpt_path,
            general_cfg=general_cfg,
            model_cfg=model_cfg,
        )
    else:
        model = SUPPORTED_MODEL[model_type](general_cfg, model_cfg)

    return model


def get_experiment_dir(root_dir, description=None):
    os.makedirs(root_dir, exist_ok=True)
    exp_nums = [int(subdir[3:]) if '_' not in subdir else int(subdir.split('_')[0][3:]) for subdir in os.listdir(root_dir)]
    max_exp_num = max(exp_nums) if len(exp_nums) > 0 else 0
    exp_name = f'exp{max_exp_num+1}' if description is None else f'exp{max_exp_num+1}_{description}'
    return os.path.join(root_dir, exp_name)



def create_paths2pos(data_dir, n_input_frames, mode='train'):
    path2pos = {}
    for jp in Path(data_dir).rglob('ball_markup.json'):
        print(f'processing {jp}')

        img_dir = str(jp.parent).replace('/annotations/', '/images/')
        data = json.load(open(jp))
        for fr in sorted(data.keys()):
            if data[fr]['x'] <= 0 or data[fr]['y'] <= 0:
                continue
            pos = data[fr]['x'] / 1920, data[fr]['y'] / 1080
            fr = int(fr)
            if fr < n_input_frames:
                continue
            ls_fr = [fr-i for i in range(n_input_frames-1, -1, -1)]  # từ nhỏ đến lớn
            ls_pos = []
            for el in ls_fr:
                if str(el) in data:
                    ls_pos.append((data[str(el)]['x']/1920, data[str(el)]['y']/1080))
                else:
                    ls_pos.append((-100, -100))

            ls_img_fp = [os.path.join(img_dir, 'img_' + '{:06d}'.format(fr) + '.jpg') for fr in ls_fr]
            ls_img_fp = [fp for fp in ls_img_fp if os.path.exists(fp)]
            if len(ls_img_fp) == n_input_frames:   # có đủ ảnh
                path2pos[tuple(ls_img_fp)] = ls_pos
    
    if mode == 'train':
        keys = list(path2pos.keys())
        np.random.seed(42)
        np.random.shuffle(keys)

        train_keys = keys[:int(0.8*len(keys))]
        val_keys = keys[int(0.8*len(keys)):]

        train_dict = {k: path2pos[k] for k in train_keys}
        val_dict = {k: path2pos[k] for k in val_keys}

        train_bin = pickle.dumps(train_dict)
        with open(f'data/gpu2_train_dict_{n_input_frames}.pkl', 'wb') as f:
            f.write(train_bin)
        
        val_bin = pickle.dumps(val_dict)
        with open(f'data/gpu2_val_dict_{n_input_frames}.pkl', 'wb') as f:
            f.write(val_bin)

    elif mode == 'test':
        test_bin = pickle.dumps(path2pos)
        with open(f'data/gpu2_test_dict_{n_input_frames}.pkl', 'wb') as f:
            f.write(test_bin)
            
    print('Done')
    return path2pos



# Define the human_readable_size function
def human_readable_size(t):
    # Get the size of A in bytes
    size_bytes = t.numel() * t.element_size()
    
    # Define the units and their multipliers
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    multipliers = [1024 ** i for i in range(len(units))]

    # Find the appropriate unit and multiplier for the input size
    for unit, multiplier in zip(units[::-1], multipliers[::-1]):
        if size_bytes >= multiplier:
            size = size_bytes / multiplier
            return f"{size:.2f} {unit}"

    # If the input size is smaller than 1 byte, return it as-is
    return f"{size_bytes} B"


def smooth_event_labelling(event_class, smooth_idx, event_frameidx):
    target_events = np.zeros((2,))
    if event_class != 2:    # 2 = empty_event
        n = smooth_idx - event_frameidx
        target_events[event_class] = np.cos(n * np.pi / 8)
        target_events[target_events < 0.01] = 0.
    return tuple(target_events)


def get_events_infor(
        root_dir,
        game_list, 
        n_input_frames=9,
        mode='train',
        smooth_labeling=True,
        event_dict={
            'bounce': 0,
            'net': 1,
            'empty_event': 2
        }
    ):
    # the paper mentioned 25, but used 9 frames only
    num_frames_from_event = int((n_input_frames - 1) / 2)

    annos_dir = os.path.join(root_dir, mode, 'annotations')
    images_dir = os.path.join(root_dir, mode, 'images')
    events_infor = {}
    events_labels = []
    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')
        events_annos_path = os.path.join(annos_dir, game_name, 'events_markup.json')
        # Load ball annotations
        json_ball = open(ball_annos_path)
        ball_annos = json.load(json_ball)
        # Load events annotations
        json_events = open(events_annos_path)
        events_annos = json.load(json_events)

        for event_frameidx, event_name in events_annos.items():
            event_frameidx = int(event_frameidx)
            smooth_frame_indices = [event_frameidx]  # By default
            if (event_name != 'empty_event') and smooth_labeling:
                smooth_frame_indices = [idx for idx in range(event_frameidx - num_frames_from_event,
                                                             event_frameidx + num_frames_from_event + 1)]  # 9 indices

            for smooth_idx in smooth_frame_indices:
                sub_smooth_frame_indices = [idx for idx in range(smooth_idx - num_frames_from_event,
                                                                 smooth_idx + num_frames_from_event + 1)]
                img_path_list = []
                for sub_smooth_idx in sub_smooth_frame_indices:
                    img_path = os.path.join(images_dir, game_name, 'img_{:06d}.jpg'.format(sub_smooth_idx))
                    img_path_list.append(img_path)
                last_f_idx = smooth_idx + num_frames_from_event
                # Get ball position for the last frame in the sequence
                if str(last_f_idx) not in ball_annos.keys():
                    print('{}, smooth_idx: {} - no ball position for the frame idx {}'.format(game_name, smooth_idx, last_f_idx))
                    continue

                ls_ball_pos = [ball_annos[str(f_idx)] if str(f_idx) in ball_annos else {'x': -100, 'y': -100} for f_idx in sub_smooth_frame_indices]
                ls_ball_pos = [(pos['x']/1920, pos['y']/1080) for pos in ls_ball_pos]
                ball_position_xy = ls_ball_pos[-1]

                # Ignore the event without ball information
                if (ball_position_xy[0] < 0) or (ball_position_xy[1] < 0):
                    continue

                # Get segmentation path for the last frame in the sequence
                seg_path = os.path.join(annos_dir, game_name, 'segmentation_masks', '{}.png'.format(last_f_idx))
                if not os.path.isfile(seg_path):
                    print("smooth_idx: {} - The segmentation path {} is invalid".format(smooth_idx, seg_path))
                    continue

                event_class = event_dict[event_name]
                target_events = smooth_event_labelling(event_class, smooth_idx, event_frameidx)
                events_infor[tuple(img_path_list)] = [ls_ball_pos, target_events, event_class, seg_path]
                # Re-label if the event is neither bounce nor net hit
                if (target_events[0] == 0) and (target_events[1] == 0):
                    event_class = 2
                events_labels.append(event_class)

    return events_infor, events_labels



if __name__ == '__main__':
    np.random.seed(42)

    # data_dir = '/data2/tungtx2/datn/ttnet/dataset/test'
    # n_input_frames = 5
    # path2pos = create_paths2pos(data_dir, n_input_frames, mode='test')

    # n_input_frames = 9
    # mode = 'test'
    # ev_info, ev_labels = get_events_infor(
    #     root_dir='/data2/tungtx2/datn/ttnet/dataset/',
    #     # game_list=['game_1', 'game_2', 'game_3', 'game_4', 'game_5'],
    #     game_list=['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7'],
    #     n_input_frames=n_input_frames,
    #     mode=mode,
    #     smooth_labeling=True,
    #     event_dict={
    #         'bounce': 0,
    #         'net': 1,
    #         'empty_event': 2
    #     }
    # )

    # if mode == 'train':
    #     keys = list(ev_info.keys())
    #     np.random.shuffle(keys)

    #     train_keys = keys[:int(0.85*len(keys))]
    #     val_keys = keys[int(0.85*len(keys)):]

    #     ev_train = {k: ev_info[k] for k in train_keys}
    #     ev_val = {k: ev_info[k] for k in val_keys}

    #     train_bin = pickle.dumps(ev_train)
    #     with open(f'data/gpu2_event_train_dict_{n_input_frames}.pkl', 'wb') as f:
    #         f.write(train_bin)

    #     val_bin = pickle.dumps(ev_val)
    #     with open(f'data/gpu2_event_val_dict_{n_input_frames}.pkl', 'wb') as f:
    #         f.write(val_bin)
    # else:
    #     test_bin = pickle.dumps(ev_info)
    #     with open(f'data/gpu2_event_test_dict_{n_input_frames}.pkl', 'wb') as f:
    #         f.write(test_bin)




    with open('data/gpu2_train_dict_5.pkl', 'rb') as f:
        obj = pickle.load(f)
    pdb.set_trace()