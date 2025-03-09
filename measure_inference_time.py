
import torch
import pandas as pd

from oml import datasets as d
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from krypto.parameters import models_dict

device = "cuda"

model_names =["vits16_dino", "facenet", "resnet18", "resnet152", "resnet101", "efficientnet_b0", "mobilenet_v3_large"]
for model_name in model_names:
    model_dict = models_dict[model_name]
    model_args = model_dict["model_args"] if "model_args" in model_dict else []
    model = model_dict["model"](*model_args)
    transform = model_dict["transform"]
    model = model.to(device).eval()

    df_test = pd.read_csv("val.csv")
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    input_tensor_key = test.input_tensors_key
    loader = DataLoader(dataset=test, batch_size=1, num_workers=1, shuffle=False)
    batch= next(iter(loader))
    img = batch[input_tensor_key].to(device)

    start = time.time()
    embeddings = model(img)
    end = time.time()
    print(f"Time taken for {model_name}: {end - start} s")