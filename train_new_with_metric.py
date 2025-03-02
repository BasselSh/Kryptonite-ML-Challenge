import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler

device = "cpu"
epochs = 1


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    fix_seed(seed=0)

    model = ViTExtractor.from_pretrained("vits16_dino").to(device).train()
    transform, _ = get_transforms_for_pretrained("vits16_dino")

    df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
    train = d.ImageLabeledDataset(df_train, transform=transform)
    val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
    sampler = BalanceSampler(train.get_labels(), n_labels=16, n_instances=4)

    def train(df_train, df_val, model, criterion, optimizer, sampler, batch_size, n_epochs, saved_model='model.pt'):

        # initialize tracker for minimum validation loss
        val_loss_min = np.Inf # set initial "min" to infinity
        
        # losses and metric to save
        train_losses = []
        val_losses = []
        val_metric = dict{}
    
        for epoch in range(n_epochs):
            pbar = tqdm(DataLoader(train, batch_sampler=sampler))
            pbar.set_description(f"epoch: {epoch}/{epochs}")
            
            train_epoch_loss = []
    
            ###################
            # train the model #
            ###################
            
            model.train() # prep model for training
            for batch in pbar:
                embeddings = model(batch["input_tensors"].to(device))
                loss = criterion(embeddings, batch["labels"].to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(criterion.last_logs)
                
                # update running training loss and metric
                train_epoch_loss.append(loss.item())
    
            ######################
            # validate the model #
            ######################
            
            model.eval() # evaluation-mode
            val_epoch_loss = []
            val_epoch_metric = []
            
            with torch.no_grad():
                for batch in pbar:
                    embeddings = inference(model, val, batch_size=32, num_workers=0, verbose=True)
    
                    # proposed in baseline
                    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=10)
                    rr = AdaptiveThresholding(n_std=2).process(rr)
                    #rr.visualize(query_ids=[2, 1], dataset=val, show=True) # for train it semms useless
                    results = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))
                    for metric_name in results.keys():
                        for k, v in results[metric_name].items():
                            print(f"{metric_name}@{k}: {v.item()}")

                    val_epoch_loss.append(loss.item())  
                    val_epoch_metric.apend(results)
        
            # calculate average loss and metric per epoch
            train_losses.append(np.mean(train_epoch_loss))
            val_losses.append(np.mean(val_epoch_loss))
            val_metric[epoch] = val_epoch_metric
    
            # print training/validation statistics
            print(f'Epoch {epoch + 1}: train_loss: {train_losses[-1]}, val_loss: {val_losses[-1]}')
    
            # save model if validation loss has decreased
            if np.mean(val_epoch_loss) <= val_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                      .format(val_loss_min, np.mean(val_epoch_loss)))
                torch.save(model.state_dict(), saved_model)
                val_loss_min = np.mean(val_epoch_loss)
    
        results = {'Train_loss': train_losses, 'Val_loss': val_losses, 'Val_metric': val_metric}
        
        return results
