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
from krypto import RealFakeQuadrupletMiner, QuadrupletLoss, RealFakeTripletDataset, TrainValSampler, FaceCutOut
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

models_dict = {
    "vits16_dino": {"model": ViTExtractor.from_pretrained("vits16_dino").to(device).train(), "transform": get_transforms_for_pretrained("vits16_dino")[0]},
    "facenet": {"model": get_facenet(), "transform": get_transforms_facenet()}
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
    parser.add_argument("description", type=str)
    parser.add_argument("--model", type=str, default="vits16_dino")
    parser.add_argument("--with_pca", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--neg_real_weight", type=float, default=1)
    parser.add_argument("--neg_fake_weight", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

class Trainer:
    def __init__(self, description, model, criterion, miner, optimizer, epochs, val_batch_size, train_dataloader, val_dataloader, with_pca=False):
        self.description = description
        self.work_dir = f"{OUTPUT_DIR}/{description}"
        self.with_pca = with_pca
        self.model = model.to(device)
        self.criterion = criterion
        self.miner = miner
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir=f"{self.work_dir}/runs")
        self.epochs = epochs
        self.val_batch_size = val_batch_size
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.best_eer = 0
        self.best_model_path = f"{self.work_dir}/best_model.pth"
        self.best_model_epoch = 0
        self.iter = 0
        self.pca = PCA(n_components=2)
        self.pca_id = 0
        self.current_epoch = 0
        self.eer_plot_dir = os.path.join(self.work_dir , "eer_plots")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(f"{self.work_dir}/pca_plots", exist_ok=True)
        os.makedirs(self.eer_plot_dir, exist_ok=True)
    
    def compute_embeddings_2d_plot(self, embeddings_tensor, labels_tensor, real_fake_tensor):
        embeddings = embeddings_tensor.detach().cpu().numpy()
        labels = labels_tensor.detach().cpu().numpy()
        real_fake = real_fake_tensor.detach().cpu().numpy()
        embeddings_2d = self.pca.fit_transform(embeddings)
        embeddings_2d_normalized = (embeddings_2d - embeddings_2d.min()) / (embeddings_2d.max() - embeddings_2d.min())
        plt.clf()
        for i, label in enumerate(labels):
            plt.text(embeddings_2d_normalized[i, 0], embeddings_2d_normalized[i, 1], f"{label}_{real_fake[i]}")
        plt.savefig(f"{self.work_dir}/pca_plots/pca_plot_{self.pca_id}.png")
        plt.clf()
        only_real_embeddings = embeddings[real_fake == 0]
        only_real_embeddings_2d = self.pca.fit_transform(only_real_embeddings)
        only_real_embeddings_2d_normalized = (only_real_embeddings_2d - only_real_embeddings_2d.min()) / (only_real_embeddings_2d.max() - only_real_embeddings_2d.min())
        labels_only_real = labels[real_fake == 0]
        for i, label in enumerate(labels_only_real):
            plt.text(only_real_embeddings_2d_normalized[i, 0], only_real_embeddings_2d_normalized[i, 1], f"{label}")
        plt.savefig(f"{self.work_dir}/pca_plots/pca_plot_real_{self.pca_id}.png")
        plt.clf()
        self.pca_id += 1
    
    def train_loop(self):
        pbar = tqdm(self.train_dataloader)
        pbar.set_description(f"epoch: {self.current_epoch}/{self.epochs}")
        total_loss = 0
        self.model.train()# prep model for training
        for batch in pbar:
            images = batch["images"].to(device)
            labels = batch["labels"]
            real_fake = batch["real_fake"]
            embeddings = self.model(images)
            if self.with_pca:
                if self.iter % 50 == 0:
                    self.compute_embeddings_2d_plot(embeddings, labels, real_fake)

            anc, pos, neg_real, neg_fake = self.miner(embeddings, labels, real_fake)
            loss_dict = self.criterion(anc, pos, neg_real, neg_fake)
            total_loss = loss_dict['total_loss']
            neg_real_loss = loss_dict['neg_real_loss']
            neg_fake_loss = loss_dict['neg_fake_loss']
            self.writer.add_scalar('Loss/total', total_loss.item(), self.iter)
            self.writer.add_scalar('Loss/neg_real', neg_real_loss.item(), self.iter)
            self.writer.add_scalar('Loss/neg_fake', neg_fake_loss.item(), self.iter)
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += total_loss.item()
            self.iter += 1
        return total_loss/len(pbar)
    
    def val_loop(self):
        pbar = tqdm(self.val_dataloader)
        pbar.set_description(f"val_epoch: {self.current_epoch}/{self.epochs}")
        self.model.eval()
        all_anchors = []
        all_pos = []
        all_neg_real = []
        all_neg_fake = []
        with torch.no_grad():
            for batch in pbar:
                images = batch["images"].to(device)
                labels = batch["labels"]
                real_fake = batch["real_fake"]
                
                embeddings = self.model(images).cpu()
                
                anchors, pos, neg_real, neg_fake = self.miner(embeddings, labels, real_fake)
                all_anchors.append(anchors)
                all_pos.append(pos)
                all_neg_real.append(neg_real)
                all_neg_fake.append(neg_fake)

        all_anchors = torch.cat(all_anchors)
        all_pos = torch.cat(all_pos)
        all_neg_real = torch.cat(all_neg_real)
        all_neg_fake = torch.cat(all_neg_fake)
        sim_pos_pos_scores = F.cosine_similarity(all_anchors, all_pos).numpy()
        sim_neg_real_scores = F.cosine_similarity(all_anchors, all_neg_real).numpy()
        sim_neg_fake_scores = F.cosine_similarity(all_anchors, all_neg_fake).numpy()
        len_labels = len(all_anchors)
        pos_pos_labels = np.ones(len_labels)
        neg_real_labels = np.zeros(len_labels)
        neg_fake_labels = np.zeros(len_labels)
        pos_pos_neg_scores = np.concatenate([sim_pos_pos_scores, sim_neg_real_scores])
        pos_pos_fake_scores = np.concatenate([sim_pos_pos_scores, sim_neg_fake_scores])
        pos_pos_neg_labels = np.concatenate([pos_pos_labels, neg_real_labels])
        pos_pos_fake_labels = np.concatenate([pos_pos_labels, neg_fake_labels])
        eer_neg_real = compute_eer(pos_pos_neg_labels, pos_pos_neg_scores, name="neg_real", save_dir=self.eer_plot_dir, plot_id=self.current_epoch)
        eer_neg_fake = compute_eer(pos_pos_fake_labels, pos_pos_fake_scores, name="neg_fake", save_dir=self.eer_plot_dir, plot_id=self.current_epoch)
        return eer_neg_real, eer_neg_fake

    def train_val(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            loss = self.train_loop()
            self.writer.add_scalar('Loss/epoch_loss', loss, epoch)
            eer_neg_real, eer_neg_fake = self.val_loop()
            self.writer.add_scalar('EER/neg_real', eer_neg_real, epoch)
            self.writer.add_scalar('EER/neg_fake', eer_neg_fake, epoch)
            average_eer = (eer_neg_real + eer_neg_fake) / 2
            self.writer.add_scalar('EER/average', average_eer, epoch)
            if average_eer > self.best_eer:
                self.best_eer = average_eer
                torch.save(self.model.state_dict(), self.best_model_path)
                self.best_model_epoch = epoch
                shutil.copy(self.best_model_path, f"model.pth")
            print(f"Best EER: {self.best_eer} at epoch {self.best_model_epoch}")


def get_loaders(transform, augmentations=None):
    train_val_dataset = RealFakeTripletDataset(root_dir="data/train/images", meta_path="data/train/meta.json", transform=transform, transforms_albu=augmentations)
    sampler_train = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=8, train=True, train_ratio=0.8)
    sampler_val = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=8, train=False, train_ratio=0.8)

    train_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_train)
    val_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_val)
    return train_dataloader, val_dataloader

def train_quadruplet(description, model, transform, with_pca, val_batch_size, epochs, loss_args, optimizer, scheduler):
    # df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
    
    # query_gallery_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)
    criterion = QuadrupletLoss(**loss_args)
    miner = RealFakeQuadrupletMiner()
    

def train_quadruplet_with_augmentation(description, model, transform, with_pca, val_batch_size, epochs, loss_args, optimizer, scheduler):
    transforms_albu = A.Compose([
        A.Resize(224, 224),
        FaceCutOut(landmarks_df="points_df_with_img_index.csv", p=0.5),
        # A.ToGray(p=0.2),
        A.HorizontalFlip(p=0.5),
        ToTensorV2() # this does not convert to float32
    ])

    df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
    train_val_dataset = RealFakeTripletDataset(root_dir="data/train/images", meta_path="data/train/meta.json", transform=transform, transforms_albu=transforms_albu)
    # query_gallery_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)
    miner = RealFakeQuadrupletMiner()
    criterion = QuadrupletLoss(**loss_args)
    sampler_train = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=8, train=True, train_ratio=0.8)
    sampler_val = TrainValSampler(train_val_dataset.get_labels(), n_labels=4, n_instances=8, train=False, train_ratio=0.8)
    train_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_train, num_workers=4)
    val_dataloader = DataLoader(train_val_dataset, batch_sampler=sampler_val, num_workers=4)

    trainer = Trainer(description, model, criterion, miner, optimizer, epochs, val_batch_size, train_dataloader, val_dataloader, with_pca=with_pca)
    trainer.train_val()

if __name__ == "__main__":
    fix_seed(seed=0)
    val_batch_size = 32
    args = parse_args()
    description = args.description
    model_name = args.model
    with_pca = args.with_pca
    val_batch_size = 32
    epochs = args.epochs
    resume = args.resume
    loss_name = args.loss
    lr = args.lr
    loss_args = {}
    if "margin" in args:
        loss_args["margin"] = args.margin
    if "neg_real_weight" in args:
        loss_args["neg_real_weight"] = args.neg_real_weight
    if "neg_fake_weight" in args:
        loss_args["neg_fake_weight"] = args.neg_fake_weight
    model_path = os.path.join(OUTPUT_DIR, description, "best_model.pth")
    model_dict = models_dict[model_name]
    model = model_dict["model"]
    if resume:
        model.load_state_dict(torch.load(model_path))
    if model_name == "facenet":
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    else:
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = None
    transform = model_dict["transform"]
    criterion = losses_dict[loss_name](**loss_args)
    miner = RealFakeQuadrupletMiner()
    train_dataloader, val_dataloader = get_loaders(transform)
    trainer = Trainer(description, model, criterion, miner, optimizer, epochs, val_batch_size, train_dataloader, val_dataloader, with_pca=with_pca)
    trainer.train_val()






# def default_settings():
#     # Inizialize hyperparameters
#     args = parse_args()
#     description = args.description
#     with_pca = args.with_pca
#     val_batch_size = 32
#     epochs = 1

#     model = ViTExtractor.from_pretrained("vits16_dino").to(device).train()
#     transform, _ = get_transforms_for_pretrained("vits16_dino")

#     df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
#     train_dataset = d.ImageLabeledDataset(df_train, transform=transform)
#     val_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

#     optimizer = Adam(model.parameters(), lr=1e-4)
#     miner = AllTripletsMiner(device=device)
#     criterion = TripletLoss(margin=1.0)
#     sampler = BalanceSampler(train_dataset.get_labels(), n_labels=4, n_instances=4)

#     trainer = Trainer(description, model, criterion, miner, optimizer, epochs, val_batch_size, train_dataset, val_dataset, sampler, with_pca=with_pca)
#     trainer.train_val()





    # def default_train_loop(self, pbar):
    #     total_loss = 0
    #     self.model.train()# prep model for training
    #     for batch in pbar:
            
    #         images = batch["input_tensors"].to(device)
    #         labels = batch["labels"]
    #         embeddings = self.model(images)
    #         if self.with_pca:
    #             if self.iter % 100 == 0:
    #                 self.compute_embeddings_2d_plot_no_fake(embeddings, labels)
    #         anc, pos, neg = self.miner.sample(embeddings, labels)
    #         loss = self.criterion(anc, pos, neg)
    #         loss.backward()
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()
    #         # pbar.set_postfix(self.criterion.last_logs)
    #         self.writer.add_scalar('Loss/train', loss.item(), self.iter)
    #         total_loss += loss.item()
    #         self.iter += 1
    #     return total_loss/len(pbar)

    # def query_gallery_loop(self, epoch):
    #     self.model.eval() # evaluation-mode
    #     embeddings = inference(self.model, self.val_dataset, batch_size=self.val_batch_size, num_workers=0, verbose=True)
    #     rr = RetrievalResults.from_embeddings(embeddings, self.val_dataset, n_items=10)
    #     rr = AdaptiveThresholding(n_std=2).process(rr)
    #     fig = rr.visualize(query_ids=[2, 1], dataset=self.val_dataset, show=True)
    #     fig.savefig(f"{OUTPUT_DIR}/val_epoch_{epoch}.png")
    #     results = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))
    #     metrics = results
    #     for metric_name in metrics.keys():
    #         metric_dict = metrics[metric_name]
    #         for k, v in metric_dict.items():
    #             self.writer.add_scalar(f'Metric/{metric_name}/{k}', v, epoch)
    #     precision = metrics['precision'][5]
    #     return results

    # def compute_embeddings_2d_plot_no_fake(self, embeddings_tensor, labels_tensor):
    #     embeddings = embeddings_tensor.detach().cpu().numpy()
    #     labels = labels_tensor.detach().cpu().numpy()
    #     embeddings_2d = self.pca.fit_transform(embeddings)
    #     embeddings_2d_normalized = (embeddings_2d - embeddings_2d.min()) / (embeddings_2d.max() - embeddings_2d.min())
    #     plt.clf()
    #     for i, label in enumerate(labels):
    #         plt.text(embeddings_2d_normalized[i, 0], embeddings_2d_normalized[i, 1], f"{label}")
    #     plt.savefig(f"{self.work_dir}/pca_plots/pca_plot_{self.pca_id}.png")
    #     self.pca_id += 1