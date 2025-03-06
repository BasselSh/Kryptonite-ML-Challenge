from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from oml.functional.losses import get_reduced
from oml.utils.misc_torch import elementwise_dist

class QuadrupletLoss(Module):

    criterion_name = "quadruplet"  # for better logging

    def __init__(self, margin: Optional[float] = 0.1, neg_real_weight=1, neg_fake_weight=1, reduction: str = "mean"):
        """

        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            reduction: ``mean``, ``sum`` or ``none``

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(QuadrupletLoss, self).__init__()

        self.margin = margin
        self.reduction = reduction
        self.neg_real_weight = neg_real_weight
        self.neg_fake_weight = neg_fake_weight

    def forward(self, anchor: Tensor, positive: Tensor, negative_real: Tensor, negative_fake: Tensor) -> Tensor:

        assert anchor.shape == positive.shape == negative_real.shape == negative_fake.shape

        positive_dist = elementwise_dist(x1=anchor, x2=positive, p=2)
        negative_real_dist = elementwise_dist(x1=anchor, x2=negative_real, p=2)
        negative_fake_dist = elementwise_dist(x1=anchor, x2=negative_fake, p=2)

        if self.margin is None:
            # here is the soft version of TripletLoss without margin
            neg_real_loss = torch.log1p(torch.exp(positive_dist - self.neg_real_weight * negative_real_dist))
            neg_fake_loss = torch.log1p(torch.exp(positive_dist - self.neg_fake_weight * negative_fake_dist))
            total_loss = neg_real_loss + neg_fake_loss
        else:
            neg_real_loss = torch.relu(self.margin + positive_dist - self.neg_real_weight * negative_real_dist)
            neg_fake_loss =  torch.relu(self.margin + positive_dist - self.neg_fake_weight * negative_fake_dist)
            total_loss = neg_real_loss + neg_fake_loss

        neg_real_loss = get_reduced(neg_real_loss, reduction=self.reduction)
        neg_fake_loss = get_reduced(neg_fake_loss, reduction=self.reduction)
        total_loss = get_reduced(total_loss, reduction=self.reduction)
        loss = dict(neg_real_loss=neg_real_loss, neg_fake_loss=neg_fake_loss, total_loss=total_loss)
        return loss
