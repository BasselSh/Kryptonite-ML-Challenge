import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from krypto.metrics import compute_eer
import torch.nn.functional as F
from torchvision.utils import make_grid
from .modules import CosDistanceHead

OUTPUT_DIR = "output"

class Trainer:
    """
    Args:
        description (str): Description of the training session, which is used for naming the output directory.
        model (torch.nn.Module): The model to be trained.
        criterion (callable): Loss function used during training.
        miner (callable): Function to mine triplets or pairs for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        logger (object): Logger for reporting metrics and plots.
        with_pca (bool): Whether to use PCA for dimensionality reduction.
        embed_size (int): Size of the embedding layer. This value is used for cosine similarity classifier.
        with_cos_head (bool): Whether to use a cosine distance head.
        device (str): Device to run the training on ('cuda' or 'cpu').
    """
    def __init__(self, description,
                 model,
                 criterion,
                 miner,
                 optimizer,
                 scheduler,
                 epochs,
                 train_dataloader,
                 val_dataloader,
                 logger,
                 with_pca=False,
                 embed_size=None,
                 with_cos_head=False,
                 device='cuda'):
        self.description = description
        self.work_dir = f"{OUTPUT_DIR}/{description}"
        self.logger = logger
        self.with_pca = with_pca
        self.model = model.to(device)
        self.criterion = criterion
        self.miner = miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir=f"{self.work_dir}/runs")
        self.epochs = epochs
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self._init_defaults()
        if with_cos_head:
            n_classes = self.train_dataloader.dataset.get_n_identities()
            self.cos_head = CosDistanceHead(num_classes=n_classes, in_channels=embed_size).to(self.device)
        else:
            self.cos_head = None
    
    def _init_defaults(self):
        self.best_eer = 1
        self.best_model_epoch = 0
        self.iter = 1
        self.pca = PCA(n_components=2)
        self.pca_id = 0
        self.current_epoch = 1
        self.patience = 10
        self.patience_counter = 0
        self.aug_batch_count = 1
        self.best_model_path = f"{self.work_dir}/best_model.pth"
        self.batch_dir = os.path.join(self.work_dir, "batch_images")
        self.eer_plot_dir = os.path.join(self.work_dir , "eer_plots")
        self.pca_plot_dir = os.path.join(self.work_dir, "pca_plots")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.pca_plot_dir, exist_ok=True)
        os.makedirs(self.eer_plot_dir, exist_ok=True)
    
    def train_loop(self):
        pbar = tqdm(self.train_dataloader)
        pbar.set_description(f"epoch: {self.current_epoch}/{self.epochs}")
        total_loss = 0
        self.model.train()# prep model for training
        n_batches = 0
        for batch in pbar:
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)
            real_fake = batch["real_fake"]
            embeddings = self.model(images)
            if self.with_pca:
                if self.iter % 500 == 0 or self.iter == 1:
                    self.compute_embeddings_2d_plot(embeddings, labels, real_fake)

            anc, pos, neg_real, neg_fake = self.miner(embeddings, labels, real_fake)
            loss_dict = self.criterion(anc, pos, neg_real, neg_fake)
            if self.cos_head is not None:
                real_filter = real_fake == 0
                cos_loss = self.cos_head(embeddings[real_filter], labels[real_filter])
                loss_dict['cos_loss'] = cos_loss
            else:
                cos_loss = 0
            
            self._log_losses_and_distances(loss_dict)
            metric_loss = loss_dict['total_loss']
            loss = metric_loss + cos_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
            total_loss += loss.item()
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
                images = batch["images"].to(self.device)
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
            eer_neg_real, eer_neg_fake = self._get_eer_real_fake(all_anchors, all_pos, all_neg_real, all_neg_fake)
        return eer_neg_real, eer_neg_fake

    def train_val(self):
        for epoch in range(self.epochs):
            if epoch == 0:
                self.plot_first_batch(epoch)
            self.current_epoch = epoch + 1
            loss = self.train_loop()
            self.writer.add_scalar('epoch_loss', loss, self.current_epoch)
            eer_neg_real, eer_neg_fake = self.val_loop()
            average_eer = (eer_neg_real + eer_neg_fake) / 2
            self._log_eer(eer_neg_real, eer_neg_fake, average_eer)
            if average_eer < self.best_eer:
                self.patience_counter = 0
                self.best_eer = average_eer
                torch.save(self.model.state_dict(), self.best_model_path)
                self.best_model_epoch = self.current_epoch
            else:
                self.patience_counter += 1
                if self.patience_counter > self.patience:
                    print(f"Early stopping at epoch {self.current_epoch}")
                    break
            print(f"EER: {average_eer} at epoch {self.current_epoch}")
    
    def compute_embeddings_2d_plot(self, embeddings_tensor, labels_tensor, real_fake_tensor):
        embeddings = embeddings_tensor.detach().cpu().numpy()
        labels = labels_tensor.detach().cpu().numpy()
        real_fake = real_fake_tensor.detach().cpu().numpy()
        embeddings_normalized = self._fit_pca(embeddings)
        
        plt.clf()
        for i, label in enumerate(labels):
            plt.text(embeddings_normalized[i, 0], embeddings_normalized[i, 1], f"{label}_{real_fake[i]}")
        plt.title(f"PCA plot real vs fake {self.pca_id}")
        plt.savefig(f"{self.work_dir}/pca_plots/pca_plot_{self.pca_id}.png")

        only_real_embeddings = embeddings[real_fake == 0]
        labels_only_real = labels[real_fake == 0]
        only_real_embeddings_normalized = self._fit_pca(only_real_embeddings)

        plt.clf()
        plt.title(f"PCA plot real {self.pca_id}")
        for i, label in enumerate(labels_only_real):
            plt.text(only_real_embeddings_normalized[i, 0], only_real_embeddings_normalized[i, 1], f"{label}")
        plt.savefig(f"{self.work_dir}/pca_plots/pca_plot_real_{self.pca_id}.png")
        plt.clf()
        self.pca_id += 1
    
    def _fit_pca(self, embeddings):
        embeddings = self.pca.fit_transform(embeddings)
        embeddings_normalized = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
        return embeddings_normalized
    
    def plot_first_batch(self, epoch):
        train_dataloader_iter = iter(self.train_dataloader) # Create a new iterator
        first_batch = next(train_dataloader_iter)
        self.first_batch = first_batch
        batch_images = first_batch["images"] 
        mean=torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) 
        std=torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) 
        batch_images = batch_images * std + mean
        batch_images = torch.clip(batch_images, min=0, max=1)
        plt.clf()
        plt.title(f"augmented images epoch {epoch}")
        plt.axis("off")
        grid = make_grid(batch_images, nrow=6)
        plt.imshow(grid.permute(1, 2, 0))
        plt.savefig(f"{self.batch_dir}/batch_{epoch}_{self.aug_batch_count}.jpg")
        plt.close()
    
    def _get_eer_real_fake(self, anchor, positive, negative_real, negative_fake):
        sim_pos_pos_scores = F.cosine_similarity(anchor, positive).numpy()
        sim_neg_real_scores = F.cosine_similarity(anchor, negative_real).numpy()
        sim_neg_fake_scores = F.cosine_similarity(anchor, negative_fake).numpy()
        len_labels = len(anchor)
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
    
    def _log_losses_and_distances(self, loss_dict):
        for k, v in loss_dict.items():
            if "dist" in k:
                self.writer.add_scalar(f'Distance/{k}', v.item(), self.iter)
            else:
                self.writer.add_scalar(f'Loss/{k}', v.item(), self.iter)
    
    def _log_eer(self, eer_neg_real, eer_neg_fake, average_eer):
        self.writer.add_scalar('EER/neg_real', eer_neg_real, self.current_epoch)
        self.writer.add_scalar('EER/neg_fake', eer_neg_fake, self.current_epoch)
        self.writer.add_scalar('EER/average', average_eer, self.current_epoch)