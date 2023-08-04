import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from my_utils import *
import torch
import pytorch_lightning as pl
from PIL import Image
from turbojpeg import TurboJPEG
import albumentations as A
from typing import List, Tuple, Dict


def load_from_pickle(fp):
    with open(fp, 'rb') as f:
        bin = f.read()
        obj = pickle.loads(bin)
    return obj


def generate_heatmap(size, center, radius):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    source: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    """
    width, height = size
    x0, y0 = center
    radius_x, radius_y = radius

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:,np.newaxis]
    heatmap = np.exp(-4*np.log(2) * ((x-x0)**2/radius_x**2 + (y-y0)**2/radius_y**2))
    return heatmap


def mask_ball_in_img(img: np.array, ls_normalized_pos: List[Tuple], r: tuple):
    h, w = img.shape[:2]
    for norm_pos in ls_normalized_pos:
        pos = (int(norm_pos[0] * w), int(norm_pos[1] * h))
        img[pos[1]-r[1] : pos[1]+r[1], pos[0]-r[0] : pos[0]+r[0], :] = 0
    return img



class MultiBallDataset(Dataset):
    def __init__(self, general_cfg, transforms, mode):
        super(MultiBallDataset, self).__init__()
        self.general_cfg = general_cfg
        self.mode = mode
        self.augment = general_cfg.training.augment
        self.n_input_frames = general_cfg.data.n_input_frames
        self.n_sample_limit = general_cfg.data.n_sample_limit
        self.mask_all = general_cfg.data.mask_all
        self.input_w, self.input_h = general_cfg.data.input_size
        self.output_w, self.output_h = self.input_w // general_cfg.data.output_stride, self.input_h // general_cfg.data.output_stride
        self.hm_gaussian_std = (general_cfg.data.ball_radius[0] / general_cfg.data.output_stride, general_cfg.data.ball_radius[1] / general_cfg.data.output_stride)
        self.transforms = transforms
        self.orig_ball_radius = self.general_cfg.data.orig_ball_radius
        self.ball_radius = self.general_cfg.data.ball_radius
        self.jpeg_reader = TurboJPEG()  # improve it later (Only initialize it once)

        # add multi ball props
        self.num_paste = self.general_cfg.training.num_paste
        self.paste_region_limit = self.general_cfg.training.paste_region_limit
        self._init_paths_and_labels()
    

    def _init_paths_and_labels(self):
        if self.mode == 'train':
            data_dict_path = self.general_cfg.data.train_dict_path
        elif self.mode == 'val':
            data_dict_path = self.general_cfg.data.val_dict_path
        else:
            data_dict_path = self.general_cfg.data.test_dict_path

        data_dict = load_from_pickle(data_dict_path)
        self.ls_img_paths = sorted(data_dict.keys())[:int(self.n_sample_limit)]
        self.ls_ball_pos = [data_dict[img_paths] for img_paths in self.ls_img_paths]


    def __len__(self):
        return len(self.ls_ball_pos)
    

    def __getitem__(self, index):
        img_paths = self.ls_img_paths[index]
        ls_norm_pos = self.ls_ball_pos[index]
        is_masked = False
        num_valid_pos = len([pos for pos in ls_norm_pos if pos[0] >= 0 and pos[1] >= 0])

        # process img
        input_imgs = []
        # add_multi_ball = self.mode == 'train' and np.random.rand() < self.general_cfg.training.add_multi_ball_prob and num_valid_pos == len(ls_norm_pos)
        add_multi_ball = num_valid_pos == len(ls_norm_pos)

        # add_multi_ball = num_valid_pos == len(ls_norm_pos)
        if add_multi_ball:
            new_first_pos = []
            first_pos = None
            ls_new_norm_pos = [[ls_norm_pos[i]] for i in range(len(img_paths))]

        for img_idx, fp in enumerate(img_paths):
            with open(fp, 'rb') as in_file:
                orig_img = self.jpeg_reader.decode(in_file.read(), 0)  # already rgb images

            # add multi ball
            if add_multi_ball:
                # get ball img of this frame
                norm_pos = ls_norm_pos[img_idx]
                abs_pos = (int(norm_pos[0]*orig_img.shape[1]), int(norm_pos[1]*orig_img.shape[0]))
                cx, cy = abs_pos
                xmin, ymin, xmax, ymax = cx - self.orig_ball_radius[0], cy - self.orig_ball_radius[1], cx + self.orig_ball_radius[0], cy + self.orig_ball_radius[1]
                ball_img = orig_img[ymin:ymax, xmin:xmax]

                # paste lung tung
                if img_idx == 0:
                    first_pos = abs_pos
                    for _ in range(self.num_paste):
                        new_cx, new_cy = np.random.randint(self.paste_region_limit[0], self.paste_region_limit[2]), np.random.randint(self.paste_region_limit[1], self.paste_region_limit[3])
                        try:
                            orig_img = cv2.seamlessClone(src=ball_img, dst=orig_img, mask=None, p=(new_cx, new_cy), flags=cv2.MONOCHROME_TRANSFER)
                            # draw a red circle
                            cv2.circle(orig_img, (new_cx, new_cy), 15, (255, 0, 0), 3)
                        except Exception as e:
                            imgs = torch.zeros(size=(3*len(img_paths), self.input_h, self.input_w), dtype=torch.float32)
                            heatmap = torch.zeros(size=(self.output_h, self.output_w), dtype=torch.float32)
                            om = torch.zeros(size=(2, self.output_h, self.output_w), dtype=torch.float32)
                            return imgs, heatmap, om
                        
                            # raise e
                            # print(e)
                            # pdb.set_trace()
                        new_first_pos.append((new_cx, new_cy))
                        ls_new_norm_pos[img_idx].append((new_cx/orig_img.shape[1], new_cy/orig_img.shape[0]))
                        # print(f'ls_new_norm_pos[{img_idx}] append')

                else:
                    pos_diff = (abs_pos[0] - first_pos[0], abs_pos[1] - first_pos[1])
                    for first_pos_x, first_pos_y in new_first_pos:
                        new_cx, new_cy = first_pos_x + pos_diff[0], first_pos_y + pos_diff[1]
                        try:
                            orig_img = cv2.seamlessClone(src=ball_img, dst=orig_img, mask=None, p=(new_cx, new_cy), flags=cv2.MONOCHROME_TRANSFER)
                            cv2.circle(orig_img, (new_cx, new_cy), 15, (255, 0, 0), 3)

                        except Exception as e:
                            imgs = torch.zeros(size=(3*len(img_paths), self.input_h, self.input_w), dtype=torch.float32)
                            heatmap = torch.zeros(size=(self.output_h, self.output_w), dtype=torch.float32)
                            om = torch.zeros(size=(2, self.output_h, self.output_w), dtype=torch.float32)
                            return imgs, heatmap, om
                            
                            # # raise e
                            # print(e)
                            # pdb.set_trace()

                        ls_new_norm_pos[img_idx].append((new_cx/orig_img.shape[1], new_cy/orig_img.shape[0]))
                        # print(f'ls_new_norm_pos[{img_idx}] append')

                # save down
                Image.fromarray(orig_img).save(f'pasted.png')
                pdb.set_trace()

            resized_img = cv2.resize(orig_img, (self.input_w, self.input_h))
            input_imgs.append(resized_img)

        # if add_multi_ball:
        #     for i, img in enumerate(input_imgs):
        #         new_positions = ls_new_norm_pos[i]
        #         for norm_pos in new_positions:
        #             abs_pos = (int(norm_pos[0]*img.shape[1]), int(norm_pos[1]*img.shape[0]))
        #             img = cv2.circle(img, abs_pos, self.ball_radius[0], (255, 0, 0), 2)
        #         Image.fromarray(img).save(f'pasted_{i}.png')
        #     pdb.set_trace()

        # ls_new_norm_pos [
        #   ((x1, y1), (x2, y2), ...),
        #   ((x1, y1), (x2, y2), ...),
        #   ((x1, y1), (x2, y2), ...),
        #   ((x1, y1), (x2, y2), ...),
        #   ((x1, y1), (x2, y2), ...),
        # ]

        ls_norm_pos = ls_new_norm_pos if add_multi_ball else [[norm_pos] for norm_pos in ls_norm_pos]   # chuyểm ls_norm_pos giống dạng của ls_new_norm_pos
        ls_out_abs_pos = np.array(ls_norm_pos[-1]) * np.array([self.output_w, self.output_h])

        if self.mode == 'train' and self.augment:
            # mask ball
            if np.random.rand() < self.general_cfg.training.mask_ball_prob and num_valid_pos == len(ls_norm_pos):     # mask ball
                input_imgs = [mask_ball_in_img(img, ball_positions, r=self.general_cfg.data.mask_radius) for img, ball_positions in list(zip(input_imgs, ls_norm_pos))]
                # out_abs_x, out_abs_y = -100, -100
                ls_out_abs_pos = [(-100, -100) for _ in range(len(ls_norm_pos[-1]))]
                ls_norm_pos[-1] = [(-1, -1) for _ in range(len(ls_norm_pos[-1]))]
                is_masked = True


            if np.random.rand() < self.general_cfg.training.augment_prob and self.transforms is not None and not is_masked:       # augment
                ls_last_norm_pos = ls_norm_pos[-1]

                # check if all pos is valid
                n_valid_pos = len([pos for pos in ls_last_norm_pos if pos[0] > 0 and pos[1] > 0])
                if n_valid_pos == len(ls_last_norm_pos):    # if all pos is valid

                    input_pos = np.array(ls_last_norm_pos) * np.array([self.input_w, self.input_h])   # only transforms points on last frames

                    if self.general_cfg.data.n_input_frames == 1:
                        transformed = self.transforms(
                            image=input_imgs[0],
                            keypoints=input_pos
                        )
                    elif self.general_cfg.data.n_input_frames == 3:
                        transformed = self.transforms(
                            image=input_imgs[0],
                            image0=input_imgs[1],
                            image1=input_imgs[2],
                            keypoints=input_pos
                        )
                    elif self.general_cfg.data.n_input_frames == 5:
                        transformed = self.transforms(
                            image=input_imgs[0],
                            image0=input_imgs[1],
                            image1=input_imgs[2],
                            image2=input_imgs[3],
                            image3=input_imgs[4],
                            keypoints=input_pos
                        )

                    if is_masked or len(transformed['keypoints']) == 0:
                        ls_out_abs_pos = [(-100, -100) for _ in range(len(ls_norm_pos[-1]))]
                    elif len(transformed['keypoints']) > 0:
                        input_abs_pos = transformed['keypoints']
                        ls_out_abs_pos = (np.array(input_abs_pos) / self.general_cfg.data.output_stride).tolist()

                    transformed_imgs = [transformed[k] for k in sorted([k for k in transformed.keys() if k.startswith('image')])]


                    # # check augment is true
                    # last_img = transformed_imgs[-1].copy()
                    # for last_input_abs_pos in input_abs_pos:
                    #     last_img = cv2.circle(last_img, (int(last_input_abs_pos[0]), int(last_input_abs_pos[1])), 5, (255, 0, 0), 2)
                    # cv2.imwrite('transformed.jpg', cv2.cvtColor(last_img, cv2.COLOR_RGB2BGR))
                    # print('write transformed')
                    # pdb.set_trace()

                    transformed_imgs = np.concatenate(transformed_imgs, axis=2)
                    transformed_imgs = torch.tensor(transformed_imgs)
                else:
                    transformed_imgs = torch.tensor(np.concatenate(input_imgs, axis=2))

            else:
                transformed_imgs = torch.tensor(np.concatenate(input_imgs, axis=2))
        
        else:
            # if self.mask_all:
            #     input_imgs = [mask_ball_in_img(img, pos, r=self.general_cfg.data.mask_radius) for img, pos in list(zip(input_imgs, ls_pos))]
            #     out_abs_x, out_abs_y = -100, -100
            transformed_imgs = torch.tensor(np.concatenate(input_imgs, axis=2))


        # check augment is true
        # last_img = transformed_imgs.numpy()[:, :, -3:].copy()
        # input_abs_pos = np.array(ls_out_abs_pos) * self.general_cfg.data.output_stride
        # for temp_pos in input_abs_pos:
        #     if all(el > 0 for el in temp_pos):
        #         last_img = cv2.circle(last_img, (int(temp_pos[0]), int(temp_pos[1])), 5, (255, 0, 0), 2)
        # cv2.imwrite('transformed.jpg', cv2.cvtColor(last_img, cv2.COLOR_RGB2BGR))
        # print('write transformed')
        # pdb.set_trace()

        # normalize
        transformed_imgs = transformed_imgs.permute(2, 0, 1) / 255.

        # process pos
        ls_center = [(int(pos[0]), int(pos[1])) for pos in ls_out_abs_pos]
        current_hm = np.zeros(shape=(self.output_h, self.output_w))
        # pdb.set_trace()
        for center in ls_center:
            if all(el>0 for el in center):
                heatmap = generate_heatmap(size=(self.output_w, self.output_h), center=center, radius=self.hm_gaussian_std)
            else:
                heatmap = np.zeros(shape=(self.output_h, self.output_w))

            current_hm = np.maximum(current_hm, heatmap)
        heatmap = current_hm
        heatmap = torch.tensor(heatmap)

        # process offset map
        offset_map = torch.zeros(2, self.output_h, self.output_w)
        for out_pos in ls_out_abs_pos:
            int_x, int_y = int(out_pos[0]), int(out_pos[1])
            offset_x, offset_y = out_pos[0] - int_x, out_pos[1] - int_y
            offset_map[0, int_y, int_x] = offset_x
            offset_map[1, int_y, int_x] = offset_y

        ls_out_int_pos = [(int(pos[0]), int(pos[1])) for pos in ls_out_abs_pos]
        ls_out_int_pos = torch.tensor(ls_out_int_pos)

        ls_last_norm_pos = ls_out_int_pos / torch.tensor([self.output_w, self.output_h])
        

        # pdb.set_trace()
        return transformed_imgs, heatmap, offset_map



class MultiBallDataModule(pl.LightningDataModule):
    def __init__(self, general_cfg, augment=True):
        super(MultiBallDataModule, self).__init__()
        self.general_cfg = general_cfg
        
        if augment:
            if self.general_cfg.data.n_input_frames == 1:
                add_target=None
            if self.general_cfg.data.n_input_frames == 3:
                add_target = {'image0': 'image', 'image1': 'image'}
            elif self.general_cfg.data.n_input_frames == 5:
                add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'}

            self.transforms = A.Compose(
                A.SomeOf([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.15, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.ColorJitter(p=0.5, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, always_apply=False),
                    A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=0),
                ], n=2),
                additional_targets=add_target,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
            )
        else:
            self.transforms = None

        self.setup(stage=None)
    

    def setup(self, stage):
        self.train_ds = MultiBallDataset(self.general_cfg, transforms=self.transforms, mode='train')
        self.val_ds = MultiBallDataset(self.general_cfg, transforms=None, mode='val')


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.general_cfg.training.bs, shuffle=self.general_cfg.training.shuffle_train, num_workers=self.general_cfg.training.num_workers, pin_memory=True)


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.general_cfg.training.bs, shuffle=False, num_workers=self.general_cfg.training.num_workers, pin_memory=True)
    


if __name__ == '__main__':
    from config import general_cfg

    ds_module = MultiBallDataModule(general_cfg, augment=True)
    ds_module.setup(stage=None)

    ds_loader = ds_module.train_dataloader()
    for i, item in enumerate(ds_loader):
        # print(i)
        transformed_imgs, heatmap, offset_map = item
        if i == 100:
            break

        # imgs = transformed_imgs[0]
        # img = imgs[:3, :, :]
        # img = img.permute(1, 2, 0).numpy()
        # img = (img*255).astype(np.uint8)
        # print(img.shape)
        # Image.fromarray(img).save('a.jpg')
        # pdb.set_trace()