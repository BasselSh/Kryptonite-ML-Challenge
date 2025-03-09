import torch.nn as nn
from typing import Optional, Dict
from torch import Tensor
import torch

class CosDistanceHead(nn.Module):
    """Classification head for `Baseline++ https://arxiv.org/abs/2003.04390`_.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        temperature (float | None): Scaling factor of `cls_score`.
            Default: None.
        eps (float): Constant variable to avoid division by zero.
            Default: 0.00001.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 temperature: Optional[float] = None,
                 eps: float = 0.00001,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} ' \
                                f'must be a positive integer'

        self.in_channels = in_channels
        self.num_classes = num_classes
        if temperature is None:
            self.temperature = 2 if num_classes <= 200 else 10
        else:
            self.temperature = temperature
        self.eps = eps
        self.init_layers()
        self.loss = nn.CrossEntropyLoss()

    def init_layers(self) -> None:
        self.fc = nn.Linear(self.in_channels, self.num_classes, bias=False)
        self.fc = nn.utils.weight_norm(self.fc, name='weight', dim=0)

    def forward(self, x: Tensor, gt_label: Tensor, **kwargs) -> Dict:
        """Forward training data."""
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + self.eps)
        cls_score = self.temperature * self.fc(x_normalized)
        losses = self.loss(cls_score, gt_label)
        return losses