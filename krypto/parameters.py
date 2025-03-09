
from .preprocessing import get_facenet_transforms, get_fastface_transforms, get_default_transforms
from .architectures import FastfaceBackbone, get_resnet18, get_facenet, get_efficientnet_b0, get_mobilenet_v3_large, get_resnet152, get_resnet101
from oml.losses import TripletLoss
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from .losses import QuadrupletLoss
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim

models_dict = {
    "vits16_dino": {"model": ViTExtractor.from_pretrained,
                    "model_args": ["vits16_dino"],
                    "transform": get_transforms_for_pretrained("vits16_dino")[0],
                    "optim_args": {"lr": 1e-4},
                    "optim_fn": optim.Adam,
                    "embed_size": 384},
    "facenet": {"model": get_facenet,
                "transform": get_facenet_transforms(),
                "optim_args": {"lr": 0.001},
                "optim_fn": optim.Adam,
                "scheduler": MultiStepLR,
                "scheduler_args": {"milestones": [10, 20, 30], "gamma": 0.1,},
                "embed_size": 1792
                },
    "resnet18": {"model": get_resnet18, 
                 "transform": get_default_transforms(), 
                 "optim_args": {"lr": 0.0001, "momentum": 0.9},
                 "optim_fn": optim.SGD,
                 "embed_size": 512},
    "resnet101": {"model": get_resnet101, 
                 "transform": get_default_transforms(), 
                 "optim_args": {"lr": 0.0001, "momentum": 0.9},
                 "optim_fn": optim.SGD,
                 "embed_size": 2048},
    "resnet152": {"model": get_resnet152, 
                 "transform": get_default_transforms(), 
                 "optim_args": {"lr": 0.0001, "momentum": 0.9},
                 "optim_fn": optim.SGD,
                 "embed_size": 2048},
    "fastface": {"model": FastfaceBackbone,
                 "transform": get_fastface_transforms(),
                 "optim_args": {"lr": 0.0001, "momentum": 0.9},
                 "optim_fn": optim.SGD,
                 "embed_size": 128},
    "efficientnet_b0": {"model": get_efficientnet_b0,
                        "transform": get_default_transforms(),
                        "optim_args": {"lr": 0.0001, "momentum": 0.9},
                        "optim_fn": optim.SGD,
                        "embed_size": 1280},
    "mobilenet_v3_large": {"model": get_mobilenet_v3_large,
                        "transform": get_default_transforms(),
                        "optim_args": {"lr": 0.0001, "momentum": 0.9},
                        "optim_fn": optim.SGD,
                        "embed_size": 960}
}
losses_dict = {
    "quadruplet": QuadrupletLoss,
    "triplet": TripletLoss,
}