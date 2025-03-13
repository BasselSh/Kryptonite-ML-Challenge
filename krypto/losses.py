from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch import Tensor, cdist

def get_reduced(tensor: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return tensor.mean()

    elif reduction == "sum":
        return tensor.sum()

    elif reduction == "none":
        return tensor

    else:
        raise ValueError(f"Unexpected type of reduction: {reduction}")

def elementwise_dist(x1: Tensor, x2: Tensor, p: int = 2) -> Tensor:
    """
    Args:
        x1: tensor with the shape of [N, D]
        x2: tensor with the shape of [N, D]
        p: degree

    Returns: elementwise distances with the shape of [N]

    """
    assert len(x1.shape) == len(x2.shape) == 2
    assert x1.shape == x2.shape

    # we need an extra dim here to avoid pairwise behavior of torch.cdist
    if len(x1.shape) == 2:
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

    dist = cdist(x1=x1, x2=x2, p=p).view(len(x1))

    return dist

class QuadrupletLoss(Module):
    """
    Quadruplet Loss for training models with quadruplet samples.

    This loss function is designed to minimize the distance between an anchor and a positive sample,
    while maximizing the distance to negative samples (both real and fake).

    Args:
        margin (Optional[float]): Margin value, set ``None`` to use `SoftTripletLoss`.
        no_fake_loss (bool): If True, the fake loss will not be included in the total loss.
        reduction (str): Specifies the reduction method to apply to the output. Options are 'mean', 'sum', or 'none'.
        lambda_fake (float): regularization factor for the fake loss.
    """
    criterion_name = "quadruplet"  # for better logging

    def __init__(self, margin: float = 1, no_fake_loss: bool = False, reduction: str = "mean", lambda_fake: float = 1, **kwargs):
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(QuadrupletLoss, self).__init__()

        self.margin = margin
        self.reduction = reduction
        self.no_fake_loss = no_fake_loss
        self.lambda_fake = lambda_fake

    def forward(self, anchor: Tensor, positive: Tensor, negative_real: Tensor, negative_fake: Tensor=None) -> Tensor:

        assert anchor.shape == positive.shape == negative_real.shape

        positive_dist = elementwise_dist(x1=anchor, x2=positive, p=2)
        negative_real_dist = elementwise_dist(x1=anchor, x2=negative_real, p=2)
        if negative_fake is not None:
            negative_fake_dist = elementwise_dist(x1=anchor, x2=negative_fake, p=2)
            self.no_fake_loss = True

        if self.margin is None:
            # here is the soft version of TripletLoss without margin
            neg_real_loss = torch.log1p(torch.exp(positive_dist - negative_real_dist))
            if not self.no_fake_loss:
                neg_fake_loss = torch.log1p(torch.exp(positive_dist - negative_fake_dist))
            else:
                neg_fake_loss = torch.tensor(0, dtype=torch.float32)
            total_loss = neg_real_loss + neg_fake_loss
        else:
            neg_real_loss = torch.relu(self.margin + positive_dist - negative_real_dist)
            if not self.no_fake_loss:
                neg_fake_loss =  self.lambda_fake * torch.relu(self.margin + positive_dist -negative_fake_dist)
            else:
                neg_fake_loss = torch.tensor(0, dtype=torch.float32)
            total_loss = neg_real_loss + neg_fake_loss

        neg_real_loss = get_reduced(neg_real_loss, reduction=self.reduction)
        neg_fake_loss = get_reduced(neg_fake_loss, reduction=self.reduction)
        total_loss = get_reduced(total_loss, reduction=self.reduction)
        positive_dist = get_reduced(positive_dist, reduction="mean")
        negative_real_dist = get_reduced(negative_real_dist, reduction="mean")
        negative_fake_dist = get_reduced(negative_fake_dist, reduction="mean")
        loss = dict(neg_real_loss=neg_real_loss, neg_fake_loss=neg_fake_loss, total_loss=total_loss, dist_pos=positive_dist, dist_neg_real=negative_real_dist, dist_neg_fake=negative_fake_dist)
        return loss


class OnlyRealLoss(Module):

    criterion_name = "only_real_loss"  # for better logging

    def __init__(self, reduction: str = "mean", **kwargs):
        assert reduction in ("mean", "sum", "none")

        super(OnlyRealLoss, self).__init__()

        self.reduction = reduction

    def forward(self, anchor: Tensor, real: Tensor) -> Tensor:
        real_dist = elementwise_dist(x1=anchor, x2=real, p=2)
        real_loss = torch.log1p(torch.exp(real_dist))
        real_loss = get_reduced(real_loss, reduction=self.reduction)
        return real_loss
