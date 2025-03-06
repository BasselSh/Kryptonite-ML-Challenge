import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import PIL
from torchvision.transforms.functional import to_pil_image
import random

class RealFakeTripletDataset(Dataset):
    def __init__(self, root_dir, meta_path, transform=None, transforms_albu=None):
        """
        Custom dataset for triplet mining with real vs. fake tracking.
        """
        self.transforms_albu = transforms_albu
        self.root_dir = root_dir
        if not transform:
            self.transform = ToTensor()
        else:
            self.transform = transform
        
        # Load metadata (real/fake annotations)
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.image_paths = []
        self.labels = []
        self.real_fake_labels = []
        all_identities = os.listdir(self.root_dir)
        # Organize images based on real/fake labels
        for identity in all_identities:  
            person_dir = os.path.join(self.root_dir, identity)
            images = sorted(os.listdir(person_dir))  # All images of this identity

            for img in images:
                img_path = os.path.join(person_dir, img)
                img_name = os.path.basename(img_path)
                img_index = os.path.join(identity, img_name)
                self.image_paths.append(img_path)
                self.labels.append(int(identity))  # Identity label
                self.real_fake_labels.append(self.meta[img_index])  # "real" or "fake"

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return self.labels
    # def _meta_to_table(self, meta):
    #     meta_dict = json.load(meta)
    #     index_rows = meta_dict.keys()
    #     real_fake_rows = meta_dict.values()
    #     combined_rows_np = np.array(list(zip(index_rows, real_fake_rows)))
    #     df = pd.DataFrame(combined_rows_np, columns=['index', 'real_fake'])
    #     print(df)
    #     return df

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        real_fake = self.real_fake_labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.transforms_albu:
            img_index = os.path.join(str(label), os.path.basename(img_path))
            if isinstance(img, torch.Tensor):
                img = np.array(to_pil_image(img))
            elif isinstance(img, PIL.Image.Image):
                img = np.array(img)
            img = self.transforms_albu(image=img, img_index=img_index)["image"]

        return {"images": img, "labels": label, "real_fake": real_fake}
