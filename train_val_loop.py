import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

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
from krypto import RealFakeQuadrupletMiner, QuadrupletLoss, RealFakeTripletDataset
device = 'cuda'
OUTPUT_DIR = "output"

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, model, criterion, miner, optimizer, epochs, val_batch_size, train_dataset, val_dataset, sampler):
        self.model = model.to('cuda')
        self.criterion = criterion
        self.miner = miner
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir=f"{OUTPUT_DIR}/runs")
        self.epochs = epochs
        self.val_batch_size = val_batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.sampler = sampler
        self.best_precision = 0
        self.best_model_path = f"{OUTPUT_DIR}/best_model.pth"
        self.best_model_epoch = 0
        self.iter = 0

    def train_loop(self, pbar):
        total_loss = 0
        self.model.train()# prep model for training
        for batch in pbar:
            self.iter += 1
            images = batch["images"].to(device) if isinstance(self.train_dataset, RealFakeTripletDataset) else batch["input_tensors"].to(device)
            embeddings = self.model(images)
            if isinstance(self.miner, RealFakeQuadrupletMiner):
                anc, pos, neg_real, neg_fake = self.miner(embeddings, batch["labels"], batch["real_fake"])
                loss = self.criterion(anc, pos, neg_real, neg_fake)
            else:
                anc, pos, neg = self.miner.sample(embeddings, batch["labels"])
                loss = self.criterion(anc, pos, neg)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # pbar.set_postfix(self.criterion.last_logs)
            self.writer.add_scalar('Loss/train', loss.item(), self.iter)
            total_loss += loss.item()
        return total_loss/len(pbar)

    def val_loop(self, epoch):
        self.model.eval() # evaluation-mode
        embeddings = inference(self.model, self.val_dataset, batch_size=self.val_batch_size, num_workers=0, verbose=True)
        rr = RetrievalResults.from_embeddings(embeddings, self.val_dataset, n_items=10)
        rr = AdaptiveThresholding(n_std=2).process(rr)
        fig = rr.visualize(query_ids=[2, 1], dataset=self.val_dataset, show=True)
        fig.savefig(f"{OUTPUT_DIR}/val_epoch_{epoch}.png")
        results = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))
        return results

    def train_val(self):
        for epoch in range(self.epochs):
            pbar = tqdm(DataLoader(self.train_dataset, batch_sampler=self.sampler))
            pbar.set_description(f"epoch: {epoch}/{self.epochs}")

            ###################
            # train the model #
            ###################
            loss = self.train_loop(pbar)
            self.writer.add_scalar('Loss/train_epoch', loss, epoch)
            ######################
            # validate the model #
            ######################
            metrics = self.val_loop(epoch)
            for metric_name in metrics.keys():
                self.writer.add_scalar(f'Metric/{metric_name}', metrics[metric_name], epoch)
            precision = metrics['precision']
            if precision > self.best_precision:
                self.best_precision = precision
                torch.save(self.model.state_dict(), self.best_model_path)
                self.best_model_epoch = epoch
            print(f"Best precision: {self.best_precision} at epoch {self.best_model_epoch}")



def default_settings():
    # Inizialize hyperparameters
    val_batch_size = 32
    epochs = 1

    model = ViTExtractor.from_pretrained("vits16_dino").to(device).train()
    transform, _ = get_transforms_for_pretrained("vits16_dino")

    df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
    train_dataset = d.ImageLabeledDataset(df_train, transform=transform)
    val_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    miner = AllTripletsMiner(device=device)
    criterion = TripletLoss(margin=1.0)
    sampler = BalanceSampler(train_dataset.get_labels(), n_labels=4, n_instances=4)

    trainer = Trainer(model, criterion, miner, optimizer, epochs, val_batch_size, train_dataset, val_dataset, sampler)
    trainer.train_val()

def quadruplet_settings():
    val_batch_size = 32
    epochs = 10

    model = ViTExtractor.from_pretrained("vits16_dino").to(device).train()
    transform, _ = get_transforms_for_pretrained("vits16_dino")

    df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
    train_dataset = RealFakeTripletDataset(root_dir="data/train/images", meta_path="data/train/meta.json", transform=transform)
    val_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    miner = RealFakeQuadrupletMiner(device=device)
    criterion = QuadrupletLoss(margin=0.1)
    sampler = BalanceSampler(train_dataset.get_labels(), n_labels=4, n_instances=6)

    trainer = Trainer(model, criterion, miner, optimizer, epochs, val_batch_size, train_dataset, val_dataset, sampler)
    trainer.train_val()

if __name__ == "__main__":
    fix_seed(seed=0)
    quadruplet_settings()
    # default_settings()
