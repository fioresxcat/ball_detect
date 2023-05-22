import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from my_utils import *
import torch
import pytorch_lightning as pl
from PIL import Image
from config import general_cfg
import time
from turbojpeg import TurboJPEG
import albumentations as A


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


def mask_ball_in_img(img: np.array, normalized_pos: tuple, r: tuple):
    h, w = img.shape[:2]
    pos = (int(normalized_pos[0] * w), int(normalized_pos[1] * h))
    img[pos[1]-r[1] : pos[1]+r[1], pos[0]-r[0] : pos[0]+r[0], :] = 0
    return img



class BallDatasetEvent(Dataset):
    def __init__(self, general_cfg, transforms, mode):
        super(BallDatasetEvent, self).__init__()
        self.general_cfg = general_cfg
        self.mode = mode
        self.n_input_frames = general_cfg.data.n_input_frames
        self.n_sample_limit = general_cfg.data.n_sample_limit
        self.input_w, self.input_h = general_cfg.data.input_size
        self.output_w, self.output_h = self.input_w // general_cfg.data.output_stride, self.input_h // general_cfg.data.output_stride
        self.hm_gaussian_std = (general_cfg.data.ball_radius[0] / general_cfg.data.output_stride, general_cfg.data.ball_radius[1] / general_cfg.data.output_stride)
        self.transforms = transforms
        self.jpeg_reader = TurboJPEG()  # improve it later (Only initialize it once)

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
        self.ls_labels = [data_dict[img_paths] for img_paths in self.ls_img_paths]
        # pdb.set_trace()


    def __len__(self):
        return len(self.ls_img_paths)
    

    def __getitem__(self, index):
        img_paths = self.ls_img_paths[index]
        ls_pos, event_target, event_class, seg_path = self.ls_labels[index]
        norm_pos = ls_pos[-1]
        abs_x, abs_y = norm_pos[0] * self.output_w, norm_pos[1] * self.output_h

        # process img
        input_imgs = []
        for fp in img_paths:
            with open(fp, 'rb') as in_file:
                # already rgb images
                resized_img = cv2.resize(self.jpeg_reader.decode(in_file.read(), 0), (self.input_w, self.input_h)) 
                input_imgs.append(resized_img)
        

        is_masked = False
        if self.mode == 'train':
            num_valid_pos = len([pos for pos in ls_pos if pos[0] >= 0 and pos[1] >= 0])
            if np.random.rand() < self.general_cfg.training.mask_ball_prob and num_valid_pos == len(ls_pos):     # mask ball
                input_imgs = [mask_ball_in_img(img, pos, r=self.general_cfg.data.mask_radius) for img, pos in list(zip(input_imgs, ls_pos))]
                abs_x, abs_y = -100, -100
                event_target = (0., 0.)
                is_masked = True

            if np.random.rand() < self.general_cfg.training.augment_prob:       # augment
                input_pos = (norm_pos[0] * self.input_w, norm_pos[1] * self.input_h)

                if self.general_cfg.data.n_input_frames == 3:
                    transformed = self.transforms(
                        image=input_imgs[0],
                        image0=input_imgs[1],
                        image1=input_imgs[2],
                        keypoints=[input_pos]
                    )
                elif self.general_cfg.data.n_input_frames == 5:
                    transformed = self.transforms(
                        image=input_imgs[0],
                        image0=input_imgs[1],
                        image1=input_imgs[2],
                        image2=input_imgs[3],
                        image3=input_imgs[4],
                        keypoints=[input_pos]
                    )
                elif self.general_cfg.data.n_input_frames == 9:
                    transformed = self.transforms(
                        image=input_imgs[0],
                        image0=input_imgs[1],
                        image1=input_imgs[2],
                        image2=input_imgs[3],
                        image3=input_imgs[4],
                        image4=input_imgs[5],
                        image5=input_imgs[6],
                        image6=input_imgs[7],
                        image7=input_imgs[8],
                        keypoints=[input_pos]
                    )

                if is_masked or len(transformed['keypoints']) == 0:
                    abs_x, abs_y = -100, -100
                    event_target = (0., 0.)
                elif len(transformed['keypoints']) > 0:
                    orig_abs_x, orig_abs_y = transformed['keypoints'][0]
                    abs_x, abs_y = orig_abs_x / self.general_cfg.data.output_stride, orig_abs_y / self.general_cfg.data.output_stride

                transformed_imgs = [transformed[k] for k in sorted([k for k in transformed.keys() if k.startswith('image')])]
                transformed_imgs = np.concatenate(transformed_imgs, axis=2)
                transformed_imgs = torch.tensor(transformed_imgs)

            else:
                transformed_imgs = torch.tensor(np.concatenate(input_imgs, axis=2))
        
        else:
            transformed_imgs = torch.tensor(np.concatenate(input_imgs, axis=2))

        # normalize
        transformed_imgs = transformed_imgs.permute(2, 0, 1) / 255.

        # process pos
        int_x, int_y = int(abs_x), int(abs_y)
        offset_x, offset_y = abs_x - int_x, abs_y - int_y

        heatmap = generate_heatmap(size=(self.output_w, self.output_h), center=(int_x, int_y), radius=self.hm_gaussian_std)
        heatmap = torch.tensor(heatmap)

        offset_map = torch.zeros(2, self.output_h, self.output_w)
        offset_map[0, int_y, int_x] = offset_x
        offset_map[1, int_y, int_x] = offset_y

        out_pos = torch.tensor([int_x, int_y])

        return transformed_imgs, heatmap, offset_map, out_pos, torch.tensor(norm_pos), torch.tensor(event_target, dtype=torch.float)



class BallDataEventModule(pl.LightningDataModule):
    def __init__(self, general_cfg, augment=True):
        super(BallDataEventModule, self).__init__()
        self.general_cfg = general_cfg
        
        if augment:
            if self.general_cfg.data.n_input_frames == 3:
                add_target = {'image0': 'image', 'image1': 'image'}
            elif self.general_cfg.data.n_input_frames == 5:
                add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'}
            elif self.general_cfg.data.n_input_frames == 9:
                add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image', 'image7': 'image'}
            

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
        self.train_ds = BallDatasetEvent(self.general_cfg, transforms=self.transforms, mode='train')
        self.val_ds = BallDatasetEvent(self.general_cfg, transforms=None, mode='val')


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds, 
            batch_size=self.general_cfg.training.bs, 
            shuffle=general_cfg.training.shuffle_train, 
            num_workers=self.general_cfg.training.num_workers,
            pin_memory=True
        )


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds, 
            batch_size=self.general_cfg.training.bs, 
            shuffle=False, 
            num_workers=self.general_cfg.training.num_workers,
            pin_memory=True
        )
    


if __name__ == '__main__':
    # transforms = A.Compose(
    #     A.SomeOf([
    #         A.HorizontalFlip(p=1),
    #         # A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.15, rotate_limit=0),
    #         # A.SafeRotate(p=0.5, limit=10, border_mode=cv2.BORDER_CONSTANT, value=0),
    #         # A.RandomBrightnessContrast(p=0.5),
    #     ], n=1),
    #     additional_targets={'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'},
    #     keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
    # )
    transforms = None
    ds = BallDatasetEvent(general_cfg, transforms, 'train')
    ls_img_fp = ds.ls_img_paths[0]
    pos = ds.ls_labels[0]
    for img_fp in ls_img_fp:
        img = cv2.imread(img_fp)
        img = cv2.resize(img, (512, 512))
        masked = mask_ball_in_img(img, pos, (10, 10))
        # abs_pos = (int(pos[0] * img.shape[1]), int(pos[1]*img.shape[0]))
        # masked = cv2.circle(img, abs_pos, 5, (0, 0, 255), thickness=1)
        cv2.imwrite('masked.jpg', masked)
        pdb.set_trace()
