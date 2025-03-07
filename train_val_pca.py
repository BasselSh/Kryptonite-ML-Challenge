import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import shutil
import os
from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLoss
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler
from torch.utils.tensorboard import SummaryWriter
from krypto import RealFakeQuadrupletMiner, QuadrupletLoss, RealFakeTripletDataset, TrainValSampler, FaceCutOut, Trainer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from eer import compute_eer
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from facenet_pytorch import fixed_image_standardization
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from clearml import Task
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
device = 'cuda'
OUTPUT_DIR = "output"

def get_facenet():
    model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=10
    ).to(device)
    model.dropout = nn.Identity()
    model.last_linear = nn.Identity()
    model.last_bn = nn.Identity()
    model.logits = nn.Identity()
    return model

def get_transforms_facenet():
    return transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

def get_resnet18():
    model =  resnet18(weights=ResNet18_Weights.DEFAULT).to(device).train()
    model.fc = nn.Identity()
    return model

models_dict = {
    "vits16_dino": {"model": ViTExtractor.from_pretrained("vits16_dino").to(device).train(),
                    "transform": get_transforms_for_pretrained("vits16_dino")[0],
                    "lr": 1e-4,
                    "embed_size": 384},
    "facenet": {"model": get_facenet(),
                "transform": get_transforms_facenet(),
                "lr": 0.001,
                "optim_args": {"lr": 0.001},
                "optim_fn": optim.Adam,
                "scheduler": MultiStepLR,
                "scheduler_args": {"milestones": [10, 20, 30], "gamma": 0.1,},
                "embed_size": 1792
                },
    "resnet18": {"model": get_resnet18(), 
                 "transform": transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
                 "optim_args": {"lr": 0.001, "momentum": 0.9},
                 "optim_fn": optim.SGD,
                 "embed_size": 512}
}
losses_dict = {
    "quadruplet": QuadrupletLoss,
    "triplet": TripletLoss
}

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--with_pca", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--loss", type=str, default="quadruplet")
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--neg_real_weight", type=float, default=1)
    parser.add_argument("--neg_fake_weight", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cutout", type=float, default=0)
    parser.add_argument("--gray", type=float, default=0)
    parser.add_argument("--no_fake_loss", action="store_true")
    parser.add_argument("--stop_fake_loss", action="store_true")
    parser.add_argument("--cutout_option", type=str, default="all")
    parser.add_argument("--with_cos_head", action="store_true")
    return parser.parse_args()

def get_loaders(transform, augmentations=None):
    train_val_dataset = RealFakeTripletDataset(root_dir="data/train/images", meta_path="data/train/meta.json", transform=transform, transforms_albu=augmentations)
    sampler_train = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=9, train=True, train_ratio=0.8)
    sampler_val = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=9, train=False, train_ratio=0.8)

    train_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_train)
    val_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_val)
    return train_dataloader, val_dataloader

def get_augmentation(cutout, cutout_option, gray):
    transforms_albu = A.Compose([
        A.Resize(224, 224),
        FaceCutOut(landmarks_df="points_df_with_img_index.csv", p=cutout, option=cutout_option),
        A.ToGray(p=gray),
        A.HorizontalFlip(p=0.5),
    ])
    return transforms_albu

if __name__ == "__main__":
    fix_seed(seed=0)
    val_batch_size = 32
    args = parse_args()
    model_name = args.model
    
    with_pca = args.with_pca
    val_batch_size = 32
    epochs = args.epochs
    resume = args.resume
    loss_name = args.loss
    no_fake_loss = args.no_fake_loss
    description = f"{model_name}_{loss_name}"
    stop_fake_loss = args.stop_fake_loss
    loss_args = {}
    if "margin" in args:
        loss_args["margin"] = args.margin
    if "neg_real_weight" in args:
        loss_args["neg_real_weight"] = args.neg_real_weight
    if "neg_fake_weight" in args:
        loss_args["neg_fake_weight"] = args.neg_fake_weight
    if "no_fake_loss" in args:
        loss_args["no_fake_loss"] = args.no_fake_loss
    # if loss_args:
    #     loss_args_names = "_".join([f"{k[0]}{v}" for k, v in loss_args.items()])
    #     description += f"_{loss_args_names}"
    if args.cutout:
        description += f"_cutout_{args.cutout}"
    if args.cutout_option != "all":
        description += f"_{args.cutout_option}"
    if args.gray:
        description += f"_gray_{args.gray}"
    description += f"_{args.description}"

    with_cos_head = args.with_cos_head
    description += f"_cos" if with_cos_head else ""
    # Description is ready
    augmentations = get_augmentation(cutout=args.cutout, cutout_option=args.cutout_option, gray=args.gray)
    
    last_work_dir_path = os.path.join(OUTPUT_DIR, description)
    os.makedirs(last_work_dir_path, exist_ok=True)
    last_work_dir_path_file = "last_work_dir.txt"
    with open(last_work_dir_path_file, 'w') as f:
        f.write(last_work_dir_path)

    model_path = os.path.join(last_work_dir_path, "best_model.pth")
    model_dict = models_dict[model_name]
    
    if "lr" in model_dict:
        lr = model_dict["lr"]
    else:
        lr = args.lr
    model = model_dict["model"]
    if resume:
        model.load_state_dict(torch.load(model_path))

    scheduler = None
    if "optim_args" in model_dict:
        optimizer = model_dict["optim_fn"](model.parameters(), **model_dict["optim_args"])
        if "scheduler" in model_dict:
            scheduler = model_dict["scheduler"](optimizer, **model_dict["scheduler_args"])
    else:
        optimizer = Adam(model.parameters(), lr=lr)

    transform = model_dict["transform"]
    embed_size = model_dict["embed_size"]
    
    criterion = losses_dict[loss_name](**loss_args)
    miner = RealFakeQuadrupletMiner()
    train_dataloader, val_dataloader = get_loaders(transform, augmentations)
    task = Task.init(task_name=description, project_name="ML_challenge")
    logger = task.get_logger()
    trainer = Trainer(description,
                        model,
                        criterion,
                        miner,
                        optimizer, 
                        scheduler, 
                        epochs, 
                        val_batch_size, 
                        train_dataloader, 
                        val_dataloader, 
                        logger, 
                        with_pca=with_pca, 
                        stop_fake_loss=stop_fake_loss, 
                        embed_size=embed_size, 
                        with_cos_head=with_cos_head)
    trainer.train_val()
