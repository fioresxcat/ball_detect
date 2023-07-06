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
from models.event_cls import *

SUPPORTED_MODEL = {
    'effunet': EffUnet,
    'effsmpunet': EffSmpUnet,
    'smpunet': SmpUnet,
    'smpunet_modified': SmpUnetModified,
    'smpdeeplab': SmpDeepLab,
    'centernet': CenterNetHourGlass,
    'centernet_yolo': CenterNetYolo,
    'centernet_yolo_multi_ball': CenterNetYoloMultiBall,
    'centernet_yolo_event': CenterNetYoloEvent,
    'my_event_cls': EventClassifier
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


def load_state_dict_for_only_bounce_model(
        new_model,
        ckpt_dir='ckpt/exp_52_ep_106',
    ):

    new_model.model.backbone.load_state_dict(torch.load(os.path.join(ckpt_dir, 'backbone.pt')), strict=True)
    new_model.model.neck.load_state_dict(torch.load(os.path.join(ckpt_dir, 'neck.pt')), strict=True)
    new_model.model.head.conv1.load_state_dict(torch.load(os.path.join(ckpt_dir, 'head_conv1.pt')), strict=True)
    new_model.model.head.conv2.load_state_dict(torch.load(os.path.join(ckpt_dir, 'head_conv2.pt')), strict=True)
    new_model.model.head.conv3.load_state_dict(torch.load(os.path.join(ckpt_dir, 'head_conv3.pt')), strict=True)
    new_model.model.head.hm_out.load_state_dict(torch.load(os.path.join(ckpt_dir, 'head_hm_out.pt')), strict=True)
    new_model.model.head.reg_out.load_state_dict(torch.load(os.path.join(ckpt_dir, 'head_reg_out.pt')), strict=True)

    return new_model


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
        with open(f'data/gpu2_train_dict_{n_input_frames}_full.pkl', 'wb') as f:
            f.write(train_bin)
        
        val_bin = pickle.dumps(val_dict)
        with open(f'data/gpu2_val_dict_{n_input_frames}_full.pkl', 'wb') as f:
            f.write(val_bin)

    elif mode == 'test':
        test_bin = pickle.dumps(path2pos)
        with open(f'data/gpu2_test_dict_{n_input_frames}_full.pkl', 'wb') as f:
            f.write(test_bin)
            
    print('Done')
    return path2pos


def gen_1_frame_data_based_on_5_frame_data(pkl_fp, save_dir, split):
    with open(pkl_fp, 'rb') as f:
        data = pickle.load(f)

    path2pos = {}
    for img_paths, labels in data.items():
        img_fp = img_paths[-1]
        pos = labels[-1]
        path2pos[tuple([img_fp])] = tuple([pos])
    
    bin = pickle.dumps(path2pos)
    with open(f'data/gpu2_{split}_dict_1.pkl', 'wb') as f:
        f.write(bin)

    return path2pos


def gen_3_frame_data_based_on_5_frame_data(pkl_fp, save_dir, split):
    with open(pkl_fp, 'rb') as f:
        data = pickle.load(f)

    path2pos = {}
    for img_paths, labels in data.items():
        img_paths = img_paths[-3:]
        ls_pos = labels[-3:]
        path2pos[tuple(img_paths)] = tuple(ls_pos)
    
    bin = pickle.dumps(path2pos)
    with open(f'data/gpu2_{split}_dict_3.pkl', 'wb') as f:
        f.write(bin)

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


import shutil
def gen_data_for_ball_detection(pkl_fp, save_dir):
    images_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    with open(pkl_fp, 'rb') as f:
        data = pickle.load(f)

    ball_radius = 20
    cnt = 0
    for img_paths, labels in data.items():
        img_fp = Path(img_paths[-1])
        pos = labels[-1]
        game_name = img_fp.parent.name
        # pdb.set_trace()
        x_center, y_center = pos
        w = ball_radius / 1920
        h = ball_radius / 1080
        yolo_anno = f'0 {x_center} {y_center} {w} {h}'
        # save image
        out_img_fp = os.path.join(images_dir, f'{game_name}_{img_fp.stem}.jpg')
        shutil.copy(str(img_fp), out_img_fp)

        # save annotation
        out_anno_fp = os.path.join(labels_dir, f'{game_name}_{img_fp.stem}.txt')
        with open(out_anno_fp, 'w') as f:
            f.write(yolo_anno)

        cnt += 1
        print(f'Done {cnt} images')


def gen_data_for_event_cls(ev_data_fp, split):
    with open(ev_data_fp, 'rb') as f:
        ev_data = pickle.load(f)
    all_img_dict = {}
    for res_split in ['train', 'val', 'test']:
        result_fp = f'results/exp71_epoch40/{res_split}/result.json'
        result_data = json.load(open(result_fp))
        all_img_dict.update(result_data['img_dict'])
    # pdb.set_trace()
    all_img_paths = sorted(list(all_img_dict.keys()))
    final_dict = {}
    for img_paths, labels in ev_data.items():
        cnt = 0
        ls_pos = []
        for fp in img_paths:
            if fp in all_img_paths:
                pred = (all_img_dict[fp]['pred'][0]/512, all_img_dict[fp]['pred'][1]/512)
                ls_pos.append(pred)
            else:
                ls_pos.append((-1, -1))
                cnt += 1
        if cnt <= 2:
            final_dict[tuple(img_paths)] = (ls_pos, labels[1])
    
    bin = pickle.dumps(final_dict)
    out_fp = f'data/exp71_epoch40/{split}_event_new_9.pkl'
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    with open(out_fp, 'wb') as f:
        f.write(bin)


def merge_no_ball_and_ball_dict():
    n_input_frames = 3
    for split in ['train', 'val', 'test']:
        if split != 'test':
            continue

        with open(f'data/gpu2_{split}_dict_{n_input_frames}.pkl', 'rb') as f:
            ball_dict = pickle.load(f)
        with open(f'data/no_ball_{split}_{n_input_frames}.pkl', 'rb') as f:
            no_ball_dict = pickle.load(f)
        
        new_no_ball_dict = {}
        for img_paths, ls_pos in no_ball_dict.items():
            new_img_paths = [os.path.join(f'/data2/tungtx2/datn/ttnet/dataset/train/', img_fp) for img_fp in img_paths]
            new_no_ball_dict[tuple(new_img_paths)] = ls_pos
        
        pdb.set_trace()

        ball_dict.update(new_no_ball_dict)
        bin_dict = pickle.dumps(ball_dict)
        with open(f'data/gpu2_{split}_dict_{n_input_frames}_add_no_ball_frames.pkl', 'wb') as f:
            f.write(bin_dict)
        print(f'Done {split}')


def merge_no_ball_and_ball_event_dict():
    for split in ['train', 'val']:
        with open(f'data/gpu2_event_{split}_dict_9.pkl', 'rb') as f:
            ball_dict = pickle.load(f)
        with open(f'data/no_ball_{split}_event.pkl', 'rb') as f:
            no_ball_dict = pickle.load(f)
        
        new_no_ball_dict = {}
        for img_paths, annos in no_ball_dict.items():
            new_img_paths = [os.path.join(f'/data2/tungtx2/datn/ttnet/dataset/train/', img_fp) for img_fp in img_paths]
            new_no_ball_dict[tuple(new_img_paths)] = annos
        
        pdb.set_trace()

        ball_dict.update(new_no_ball_dict)
        bin_dict = pickle.dumps(ball_dict)
        with open(f'data/gpu2_event_{split}_dict_9_add_no_ball_frames.pkl', 'wb') as f:
            f.write(bin_dict)
        print(f'Done {split}')



def augment_ball():
    ball_radius = 15
    paste_region_limit = (300, 300, 1500, 800)  # xmin, ymin, xmax, ymax
    num_paste = 3
    out_dir = 'augment_ball_data'
    for split in ['train', 'val', 'test']:
        with open(f'data/gpu2_{split}_dict_5.pkl', 'rb') as f:
            ball_dict = pickle.load(f)
        
        new_ball_dict = {}
        cnt = 0
        for img_paths, ls_pos in ball_dict.items():
            is_valid = True
            for pos in ls_pos:
                if any(el<0 for el in pos):
                    is_valid = False
                    break
            if not is_valid:
                continue
            new_first_pos = []
            first_pos = None
            pasted_imgs = []
            new_pos = []
            for img_idx, img_fp in enumerate(img_paths):
                # get orig img and pos
                img = cv2.imread(str(img_fp))
                norm_pos = ls_pos[img_idx]
                have_ball = norm_pos[0] > 0 and norm_pos[1] > 0
                if have_ball:
                    abs_pos = (int(norm_pos[0]*img.shape[1]), int(norm_pos[1]*img.shape[0]))
                    cx, cy = abs_pos
                    xmin, ymin, xmax, ymax = cx - ball_radius, cy - ball_radius, cx + ball_radius, cy + ball_radius
                    ball_img = img[ymin:ymax, xmin:xmax]

                # paste pos
                if img_idx == 0:
                    first_pos = abs_pos
                    for _ in range(num_paste):
                        new_cx, new_cy = np.random.randint(paste_region_limit[0], paste_region_limit[2]), np.random.randint(paste_region_limit[1], paste_region_limit[3])
                        try:
                            img = cv2.seamlessClone(src=ball_img, dst=img, mask=None, p=(new_cx, new_cy), flags=cv2.NORMAL_CLONE)
                        except Exception as e:
                            print(e)
                            pdb.set_trace()
                        new_first_pos.append((new_cx, new_cy))
                    pasted_imgs.append(img)
                else:
                    pos_diff = (abs_pos[0] - first_pos[0], abs_pos[1] - first_pos[1]) if have_ball else (0, 0)
                    for first_pos_x, first_pos_y in new_first_pos:
                        new_cx, new_cy = first_pos_x + pos_diff[0], first_pos_y + pos_diff[1]
                        try:
                            img = cv2.seamlessClone(src=ball_img, dst=img, mask=None, p=(new_cx, new_cy), flags=cv2.NORMAL_CLONE)
                        except Exception as e:
                            print(e)
                            pdb.set_trace()
                    pasted_imgs.append(img)
                

            # new_paths = []
            # for i, img in enumerate(pasted_imgs):
            #     old_fp = img_paths[i]
            #     fn = old_fp.parent.name + '_' + old_fp.name
            #     new_fp = os.path.join(out_dir, split, fn)
            #     os.makedirs(os.path.dirname(new_fp), exist_ok=True)
            #     cv2.imwrite(new_fp, img)
            #     new_paths.append(new_fp)
            # new_ball_dict[tuple(new_paths)] = new_first_pos

            cnt += 1
            if cnt == 5:
                for i, img in enumerate(pasted_imgs):
                    cv2.imwrite(f'pasted_{i}.jpg', img)
                break
        break




if __name__ == '__main__':
    np.random.seed(42)

    augment_ball()

    # split = 'train'
    # with open(f'data/gpu2_{split}_dict_3_add_no_ball_frames.pkl', 'rb') as f:
    #     ball_dict = pickle.load(f)
    # items = list(ball_dict.items())
    # print(items[100])
    # pdb.set_trace()

    # gen_data_for_event_cls('data/gpu2_event_val_dict_9.pkl', 'test')

    # merge_no_ball_and_ball_dict()

    # merge_no_ball_and_ball_event_dict()

    # for split in ['train', 'val', 'test']:
    #     ev_data_fp = f'data/gpu2_event_{split}_dict_9.pkl'
    #     gen_data_for_event_cls(ev_data_fp, split)
    
    # for split in ['train', 'val']:
    #     n_frames = 9
    #     fp = f'data/gpu2_event_{split}_dict_{n_frames}_add_no_ball_frames.pkl'
    #     with open(fp, 'rb') as f:
    #         obj = pickle.load(f)
    #     for k, v in obj.items():
    #         v = list(v)
    #         ls_pos = list(v[0])
    #         for i, el in enumerate(ls_pos):
    #             if any(t<0 for t in el):
    #                 ls_pos[i] = (-1, -1)
    #         v[0] = tuple(ls_pos)
    #         obj[k] = tuple(v)
        
    #     pdb.set_trace()
    #     obj_bin = pickle.dumps(obj)
    #     with open(fp, 'wb') as f:
    #         f.write(obj_bin)

    # with open(f'data/gpu2_event_train_dict_9.pkl', 'rb') as f:
    #     obj = pickle.load(f)
    # items = list(obj.items())
    # print(items[0])
    # print(len(items))
    # pdb.set_trace()



    # pkl_fp = 'data/gpu2_val_dict_5.pkl'
    # save_dir = '/data2/tungtx2/datn/yolov8/ball_detection_data/val'
    # gen_data_for_ball_detection(pkl_fp, save_dir)


    # data_dir = '/data2/tungtx2/datn/ttnet/dataset/train'
    # n_input_frames = 3
    # path2pos = create_paths2pos(data_dir, n_input_frames, mode='train')

    # gen_3_frame_data_based_on_5_frame_data(
    #     pkl_fp='data/gpu2_train_dict_5.pkl',
    #     save_dir='data',
    #     split='train'
    # )

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




    # with open('data/gpu2_event_val_dict_9.pkl', 'rb') as f:
    #     obj = pickle.load(f)
    # pdb.set_trace()