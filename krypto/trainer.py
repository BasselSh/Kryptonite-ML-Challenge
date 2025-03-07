
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from eer import compute_eer
import torch.nn.functional as F
import albumentations as A
import clearml
from torchvision.utils import make_grid
from krypto import CosineDistanceHead
device = 'cuda'
OUTPUT_DIR = "output"

class Trainer:
    def __init__(self, description,
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
                 with_pca=False,
                 stop_fake_loss=False,
                 embed_size=None,
                 with_cos_head=False):
        self.description = description
        self.work_dir = f"{OUTPUT_DIR}/{description}"
        self.logger = logger
        self.with_pca = with_pca
        self.stop_fake_loss = stop_fake_loss
        self.model = model.to(device)
        self.criterion = criterion
        self.miner = miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir=f"{self.work_dir}/runs")
        self.epochs = epochs
        self.val_batch_size = val_batch_size
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.best_eer = 1
        self.best_model_path = f"{self.work_dir}/best_model.pth"
        self.best_model_epoch = 0
        self.iter = 1
        self.pca = PCA(n_components=2)
        self.pca_id = 0
        self.current_epoch = 1
        self.eer_plot_dir = os.path.join(self.work_dir , "eer_plots")
        self.patience = 5
        self.patience_counter = 0
        self.aug_batch_count = 1
        self.batch_dir = os.path.join(self.work_dir, "batch_images")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(f"{self.work_dir}/pca_plots", exist_ok=True)
        os.makedirs(self.eer_plot_dir, exist_ok=True)
        if with_cos_head:
            n_classes = self.train_dataloader.dataset.get_n_identities()
            self.cos_head = CosineDistanceHead(num_classes=n_classes, in_channels=embed_size)
        else:
            self.cos_head = None
    

    def compute_embeddings_2d_plot(self, embeddings_tensor, labels_tensor, real_fake_tensor):
        embeddings = embeddings_tensor.detach().cpu().numpy()
        labels = labels_tensor.detach().cpu().numpy()
        real_fake = real_fake_tensor.detach().cpu().numpy()
        embeddings_2d = self.pca.fit_transform(embeddings)
        embeddings_2d_normalized = (embeddings_2d - embeddings_2d.min()) / (embeddings_2d.max() - embeddings_2d.min())
        plt.clf()
        for i, label in enumerate(labels):
            plt.text(embeddings_2d_normalized[i, 0], embeddings_2d_normalized[i, 1], f"{label}_{real_fake[i]}")
        plt.title(f"PCA plot real vs fake {self.pca_id}")
        plt.savefig(f"{self.work_dir}/pca_plots/pca_plot_{self.pca_id}.png")
        only_real_embeddings = embeddings[real_fake == 0]
        only_real_embeddings_2d = self.pca.fit_transform(only_real_embeddings)
        only_real_embeddings_2d_normalized = (only_real_embeddings_2d - only_real_embeddings_2d.min()) / (only_real_embeddings_2d.max() - only_real_embeddings_2d.min())
        labels_only_real = labels[real_fake == 0]

        plt.clf()
        plt.title(f"PCA plot real {self.pca_id}")
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
        n_batches = 0
        for batch in pbar:
            images = batch["images"].to(device)
            labels = batch["labels"]
            real_fake = batch["real_fake"]
            embeddings = self.model(images)
            if self.cos_head is not None:
                loss_cos = self.cos_head(embeddings, labels)
                loss_cos.backward()
                self.writer.add_scalar('Loss/cos_head', loss_cos.item(), self.iter)
            if self.with_pca:
                if self.iter % 500 == 0 or self.iter == 1:
                    self.compute_embeddings_2d_plot(embeddings, labels, real_fake)

            anc, pos, neg_real, neg_fake = self.miner(embeddings, labels, real_fake)
            loss_dict = self.criterion(anc, pos, neg_real, neg_fake)
            total_loss = loss_dict['total_loss']
            for k, v in loss_dict.items():
                if "dist" in k:
                    self.writer.add_scalar(f'Distance/{k}', v.item(), self.iter)
                else:
                    self.writer.add_scalar(f'Loss/{k}', v.item(), self.iter)
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
            total_loss += total_loss.item()
            self.iter += 1
            n_batches += 1
        return total_loss/n_batches
    
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
        eer_neg_real = compute_eer(pos_pos_neg_labels, pos_pos_neg_scores, name="neg_real", save_dir=self.eer_plot_dir, plot_id=self.current_epoch, logger=self.logger)
        eer_neg_fake = compute_eer(pos_pos_fake_labels, pos_pos_fake_scores, name="neg_fake", save_dir=self.eer_plot_dir, plot_id=self.current_epoch, logger=self.logger)
        return eer_neg_real, eer_neg_fake

    def train_val(self):
        for epoch in range(self.epochs):
            if self.train_dataloader.dataset.transforms_albu:
                for i in range(self.aug_batch_count):
                    batch_images = next(iter(self.train_dataloader))["images"]
                    plt.title("augmented images")
                    plt.axis("off")
                    grid = make_grid(batch_images, nrow=6)
                    plt.imshow(grid.permute(1, 2, 0))
                    plt.savefig(f"{self.batch_dir}/batch_{self.aug_batch_count}.jpg")
                    plt.clf()
                    self.aug_batch_count += 1
            self.current_epoch = epoch + 1
            if self.stop_fake_loss and self.current_epoch == 2:
                self.criterion.no_fake_loss = True
            loss = self.train_loop()
            self.writer.add_scalar('epoch_loss', loss, self.current_epoch)
            eer_neg_real, eer_neg_fake = self.val_loop()
            self.writer.add_scalar('EER/neg_real', eer_neg_real, self.current_epoch)
            self.writer.add_scalar('EER/neg_fake', eer_neg_fake, self.current_epoch)
            average_eer = (eer_neg_real + eer_neg_fake) / 2
            self.writer.add_scalar('EER/average', average_eer, self.current_epoch)
            if average_eer < self.best_eer:
                self.patience_counter = 0
                self.best_eer = average_eer
                torch.save(self.model.state_dict(), self.best_model_path)
                self.best_model_epoch = self.current_epoch
                shutil.copy(self.best_model_path, f"model.pth")
            else:
                self.patience_counter += 1
                if self.patience_counter > self.patience:
                    print(f"Early stopping at epoch {self.current_epoch}")
                    break
            print(f"EER: {average_eer} at epoch {self.current_epoch}")
