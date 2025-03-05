
import torch
import pandas as pd

from oml import datasets as d
from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr

from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
import time
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
import torch.nn as nn
device = "cuda"

model = ViTExtractor.from_pretrained("vits16_dino")
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device).eval()

transform, _ = get_transforms_for_pretrained("vits16_dino")

df_test = pd.read_csv("val.csv")
test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
input_tensor_key = test.input_tensors_key
loader = DataLoader(dataset=test, batch_size=1, num_workers=1, shuffle=False)
batch= next(iter(loader))
img = batch[input_tensor_key].to(device)
# model = efficientnet_b0(pretrained=False).to(device).eval()
# print(model)
# model.classifier = nn.Identity()

# print(img.shape)
start = time.time()
embeddings = model(img)
end = time.time()
print(f"Time taken: {end - start}")