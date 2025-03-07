from krypto import FaceCutOut, RealFakeTripletDataset
from torch.utils.data import DataLoader
import albumentations as A
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
transforms = A.Compose([
    A.Resize(224, 224),
    # FaceCutOut(landmarks_df="points_df_with_img_index.csv", p=0.5),
    # A.ToGray(p=0.2),
    A.HorizontalFlip(p=0.5),
])

train_dataset = RealFakeTripletDataset(root_dir="data/train/images", meta_path="data/train/meta.json", transforms_albu=transforms)

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    grid = make_grid(batch['images'], nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("transforms_test.png")
    break
