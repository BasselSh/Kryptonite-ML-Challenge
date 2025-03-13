import torch.nn as nn
import fastface as ff
from torchvision.models import resnet18, ResNet18_Weights, resnet152, ResNet152_Weights, resnet101, ResNet101_Weights
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights

class FastfaceBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ff.FaceDetector.from_pretrained("lffd_original").arch.backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.avgpool(feats[-1]).flatten(1)
        return feats
    
def get_resnet18():
    model =  resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model

def get_resnet101():
    model =  resnet101(weights=ResNet101_Weights.DEFAULT)
    model.fc = nn.Identity()
    for name, param in model.named_parameters():
        if 'layer4' in name or 'layer3' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def get_resnet152():
    model =  resnet152(weights=ResNet152_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model


def get_facenet():
    model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=10
    )
    model.dropout = nn.Identity()
    model.last_linear = nn.Identity()
    model.last_bn = nn.Identity()
    model.logits = nn.Identity()
    return model

def get_efficientnet_b0(pretrained=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier = nn.Identity()
    return model

def get_mobilenet_v3_large(pretrained=True):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    model.classifier = nn.Identity()
    return model

