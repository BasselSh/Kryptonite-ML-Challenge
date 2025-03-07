
from krypto.datasets import RealFakeTripletDataset
from krypto.sampler import TrainValSampler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time
import os
import random
import numpy as np
import torch
from krypto import RealFakeQuadrupletMiner, QuadrupletLoss, RealFakeTripletDataset, TrainValSampler, FaceCutOut
import torch
import albumentations as A

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_seed(seed=0)

transforms_albu = A.Compose([
    A.Resize(224, 224),
    FaceCutOut(landmarks_df="points_df_with_img_index.csv", p=0.5),
    A.ToGray(p=0.2),
    A.HorizontalFlip(p=0.5),
])

dataset = RealFakeTripletDataset(root_dir="data/train/images", meta_path="data/train/meta.json", transforms_albu=transforms_albu)
sampler = TrainValSampler(dataset.get_labels(), n_labels=4, n_instances=4, train=False, train_ratio=0.8)
dataloader = DataLoader(dataset, batch_sampler=sampler)
miner = RealFakeQuadrupletMiner()

# samples_dir = "output/samples"
# sub_dirs = ["pos_pos", "pos_neg_real", "pos_neg_fake"]
# os.makedirs(samples_dir, exist_ok=True)
# for sub_dir in sub_dirs:
#     os.makedirs(os.path.join(samples_dir, sub_dir), exist_ok=True)
# ii = 1
# for i, batch in enumerate(dataloader):
#     if i == ii:
#         break
#     anchors, pos, neg_real, neg_fake = miner(batch['images'], batch['labels'], batch['real_fake'])
#     pos_pos = torch.cat([anchors, pos], dim=-1)
#     figure = plt.figure()
#     plt.clf()
#     img = make_grid(pos_pos, nrow=4)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.savefig(f"{samples_dir}/{sub_dirs[0]}/test_sampler_{i}.png")
#     plt.close(figure)

#     pos_neg_real = torch.cat([anchors, neg_real], dim=-1)
#     figure = plt.figure()
#     plt.clf()
#     img = make_grid(pos_neg_real, nrow=4)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.savefig(f"{samples_dir}/{sub_dirs[1]}/test_sampler_{i}.png")
#     plt.close(figure)

#     pos_neg_fake = torch.cat([anchors, neg_fake], dim=-1)
#     figure = plt.figure()
#     plt.clf()
#     img = make_grid(pos_neg_fake, nrow=4)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.savefig(f"{samples_dir}/{sub_dirs[2]}/test_sampler_{i}.png")
#     plt.close(figure)

    
# dataloader = DataLoader(dataset, batch_sampler=sampler)
samples_dir = "output/samples_aug"
os.makedirs(samples_dir, exist_ok=True)
for i, batch in enumerate(dataloader):
    figure = plt.figure()
    plt.clf()
    img = make_grid(batch['images'], nrow=4)
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig(f"{samples_dir}/test_sampler_{i}.png")
    plt.close(figure)
    