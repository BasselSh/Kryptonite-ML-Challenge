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
    def __init__(self, model, criterion, miner, optimizer, epochs, batch_size, train_dataset, val_dataset, sampler):
        self.model = model.to('cuda')
        self.criterion = criterion
        self.miner = miner
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir=f"{OUTPUT_DIR}/runs")
        self.epochs = epochs
        self.batch_size = batch_size
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
            embeddings = self.model(batch["input_tensors"].to(device))
            anc, pos, neg = self.miner(embeddings, batch["labels"].to(device))
            loss = self.criterion(anc, pos, neg)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            pbar.set_postfix(self.criterion.last_logs)
            self.writer.add_scalar('Loss/train', loss.item(), self.iter)
            total_loss += loss.item()
        return total_loss/len(pbar)

    def val_loop(self, epoch):
        self.model.eval() # evaluation-mode
        embeddings = inference(self.model, self.val_dataset, batch_size=self.batch_size, num_workers=0, verbose=True)
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
if __name__ == "__main__":
    fix_seed(seed=0)
    
    # Inizialize hyperparameters
    batch_size = 32
    epochs = 1

    model = ViTExtractor.from_pretrained("vits16_dino").to(device).train()
    transform, _ = get_transforms_for_pretrained("vits16_dino")

    df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
    train_dataset = d.ImageLabeledDataset(df_train, transform=transform)
    val_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    miner = AllTripletsMiner(device=device)
    criterion = TripletLoss()
    sampler = BalanceSampler(train_dataset.get_labels(), n_labels=4, n_instances=4)

    trainer = Trainer(model, criterion, miner, optimizer, epochs, batch_size, train_dataset, val_dataset, sampler)
    # trainer.train_val()
    trainer.val_loop(0)

