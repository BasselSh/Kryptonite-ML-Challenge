import torch
from torch.optim import Adam
import os
from krypto.trainer import Trainer
import argparse
from clearml import Task
from krypto.utils import get_augmentation, get_loaders
from krypto.miners import RealFakeQuadrupletMiner
from krypto.parameters import models_dict, losses_dict

device = 'cuda'
OUTPUT_DIR = "output"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--no_pca", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--loss", type=str, default="quadruplet")
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0)
    parser.add_argument("--cutout", type=float, default=0)
    parser.add_argument("--gray", type=float, default=0)
    parser.add_argument("--no_fake_loss", action="store_true")
    parser.add_argument("--cutout_option", type=str, default="all")
    parser.add_argument("--with_cos_head", action="store_true")
    parser.add_argument("--lambda_fake", type=float, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    with_pca = not args.no_pca
    epochs = args.epochs
    resume = args.resume
    loss_name = args.loss
    
    description = f"{model_name}_{loss_name}"
    loss_args = {}
    loss_args["margin"] = args.margin
    no_fake_loss = args.no_fake_loss
    if no_fake_loss:
        loss_args["no_fake_loss"] = no_fake_loss
        description += "_no_fake_loss"
    if args.lambda_fake != 0:
        description += f"_lambda_fake_{args.lambda_fake}"
        loss_args["lambda_fake"] = args.lambda_fake
    if args.cutout:
        description += f"_cutout_{args.cutout}"
    if args.cutout_option != "all":
        description += f"_{args.cutout_option}"
    if args.gray:
        description += f"_gray_{args.gray}"
    description += f"_{args.description}"

    with_cos_head = args.with_cos_head
    description += f"_cos" if with_cos_head else ""
    # ..... Description is ready .....

    augmentations = get_augmentation(cutout=args.cutout, cutout_option=args.cutout_option, gray=args.gray)
    model_dict = models_dict[model_name]
    model_cls = model_dict["model"]
    if "model_args" in model_dict:
        model = model_cls(*model_dict["model_args"])
    else:
        model = model_cls()

    last_work_dir_path = os.path.join(OUTPUT_DIR, description)
    os.makedirs(last_work_dir_path, exist_ok=True)
    last_work_dir_path_file = "last_work_dir.txt"
    with open(last_work_dir_path_file, 'w') as f:
        f.write(last_work_dir_path)
    model_path = os.path.join(last_work_dir_path, "best_model.pth")
    if resume:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device).train()
    scheduler = None
    if args.lr != 0:
        lr = args.lr
        description += f"_lr_{args.lr}"
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        lr = model_dict["optim_args"]['lr']
        optimizer = model_dict["optim_fn"](model.parameters(), **model_dict["optim_args"])
        if "scheduler" in model_dict:
            scheduler = model_dict["scheduler"](optimizer, **model_dict["scheduler_args"])

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
                        train_dataloader, 
                        val_dataloader,
                        logger, 
                        with_pca=with_pca, 
                        embed_size=embed_size, 
                        with_cos_head=with_cos_head,
                        device=device)
    trainer.train_val()
