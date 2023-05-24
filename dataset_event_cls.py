import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from my_utils import *
import torch
import pytorch_lightning as pl
import albumentations as A
from train_event_cls import *


def load_from_np(fp):
    arr = np.load(fp)
    x = arr['input']
    label = arr['label']
    return x, label


class EventClassifyDataset(Dataset):
    def __init__(self, general_cfg, mode):
        super(EventClassifyDataset, self).__init__()
        self.general_cfg = general_cfg
        self.mode = mode
        self.data_dir = os.path.join(general_cfg.data.root_dir, mode)
        self.ls_sample_paths = sorted(Path(self.data_dir).glob('*.npz'))


    def __len__(self):
        return len(self.ls_sample_paths)
    

    def __getitem__(self, index):
        sample_fp = self.ls_sample_paths[index]
        x, label = load_from_np(sample_fp)  # all numpy array, shape (5, 128, 128) and (2,)
        return torch.from_numpy(x), torch.from_numpy(label)
        


class EventClassifyDataModule(pl.LightningDataModule):
    def __init__(self, general_cfg, augment=True):
        super(EventClassifyDataModule, self).__init__()
        self.general_cfg = general_cfg
        self.transforms = None
        self.setup(stage=None)
    

    def setup(self, stage):
        self.train_ds = EventClassifyDataset(self.general_cfg, mode='train')
        self.val_ds = EventClassifyDataset(self.general_cfg, mode='val')


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.general_cfg.training.bs, shuffle=self.general_cfg.training.shuffle_train, num_workers=self.general_cfg.training.num_workers, pin_memory=True)


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.general_cfg.training.bs, shuffle=False, num_workers=self.general_cfg.training.num_workers, pin_memory=True)
    


if __name__ == '__main__':
    ds = EventClassifyDataset(general_cfg, mode='train')
    for i, item in enumerate(ds):
        if i == 0:
            x, label = item
            break
    print(x.shape, label.shape)
