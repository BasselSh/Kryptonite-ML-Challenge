
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from oml.models import ViTExtractor
import torch
from facenet_pytorch import InceptionResnetV1
device = 'cuda'
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

img = torch.randn(1, 3, 224, 224).to(device)
# model = ViTExtractor.from_pretrained("vits16_dino").to(device)
# print(model(img).shape)
model = get_facenet()
print(model(img).shape)