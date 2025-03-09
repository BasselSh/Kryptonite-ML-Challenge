from torchvision.transforms.functional import to_tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from facenet_pytorch import fixed_image_standardization
from torchvision import transforms
import numpy as np

def get_facenet_transforms():
    return transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

def get_default_transforms():
    return transforms.Compose([transforms.Resize(224), 
                               transforms.ToTensor(), 
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])