
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
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from facenet_pytorch import fixed_image_standardization
from torchvision import transforms
import numpy as np

device = "cuda"


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
model_name = "facenet"
model_dict = models_dict[model_name]
model = model_dict["model"]
transform = model_dict["transform"]
model.load_state_dict(torch.load("model.pth"))
model = model.to(device).eval()

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