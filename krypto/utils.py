import random
import numpy as np
import torch
import albumentations as A
from .transforms import FaceCutOut
from .datasets import RealFakeTripletDataset
from .samplers import TrainValSampler
from torch.utils.data import DataLoader
from .paths_cfg import DATA_DIR

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_augmentation(cutout, cutout_option, gray):
    transforms_albu = A.Compose([
        A.Resize(224, 224),
        FaceCutOut(landmarks_df="points_df_with_img_index.csv", p=cutout, option=cutout_option),
        A.ToGray(p=gray),
        A.HorizontalFlip(p=0.5),
    ])
    return transforms_albu

def get_loaders(transform, augmentations=None):
    train_val_dataset = RealFakeTripletDataset(root_dir=f"{DATA_DIR}/images", meta_path="meta.json", transform=transform, transforms_albu=augmentations)
    sampler_train = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=9, train=True, train_ratio=0.8)
    sampler_val = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=9, train=False, train_ratio=0.8)
    sampler_real_only = TrainValSampler(train_val_dataset.get_labels(), n_labels=10, n_instances=1, train=True, train_ratio=0.8)
    train_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_train, shuffle=False)
    
    val_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_val, shuffle=False)
    return train_dataloader, val_dataloader
