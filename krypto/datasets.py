import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np

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
        self._num_labels = len(all_identities)
        self.all_indecies = []
        # Organize images based on real/fake labels
        for identity in all_identities:  
            person_dir = os.path.join(self.root_dir, identity)
            images = sorted(os.listdir(person_dir))  # All images of this identity

            for img in images:
                img_path = os.path.join(person_dir, img)
                img_name = os.path.basename(img_path)
                img_index = os.path.join(str(int(identity)), img_name)
                self.all_indecies.append(img_index)
                self.image_paths.append(img_path)
                self.labels.append(int(identity))  # Identity label
                self.real_fake_labels.append(self.meta[img_index])  # "real" or "fake"

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return self.labels
    
    def get_n_identities(self):
        return self._num_labels
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        real_fake = self.real_fake_labels[idx]
        img_index = self.all_indecies[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms_albu:
            img = np.array(img)
            img = self.transforms_albu(image=img, img_index=img_index)["image"]
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return {"images": img, "labels": label, "real_fake": real_fake, "img_index": img_index}
