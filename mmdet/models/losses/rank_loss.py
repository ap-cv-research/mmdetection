from typing import Callable
import torch
import torch.nn as nn
# from ..builder import LOSSES
# from .utils import weight_reduce_loss


__all__ = ('RankingLoss',)


def _gt_mask(labels: torch.Tensor) -> torch.Tensor:
    target = torch.unsqueeze(labels,0)
    mask = target == torch.transpose(target, dim0=1, dim1=0)
    return mask


class CosineDistance(object):

    _s: Callable

    def __init__(self, dim=2, eps=1e-6):

        self._s = nn.CosineSimilarity(dim=dim, eps=eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        
        c = tensor.shape[-1]
        t = tensor.view(-1, 1, c)
        similarity = self._s(t, torch.transpose(t, dim0=1, dim1=0))
        
        return 0.5 - 0.5 * similarity


def triplet_batch_all(mask: torch.Tensor, dist: torch.Tensor, margin: float) -> torch.Tensor:
    """Batch-All triplet mining stragegy.

    Triplet version of the ranking loss seems to outperform the pair-wise
    version. The triplets are formed by an anchor sample x_a, positive
    sample x_p and a negative sample x_n. The objective is that the
    distance between the anchor sample and the negative sample d(r_a,
    r_n) representations is greater (and bigger than a margin m) than the
    distance between the anchor and positive representations d(r_a, r_p):

    L(ra, rp, rn) = max(0, m + d(r_a, r_p) - d(r_a, r_n))

    E.g.:

        Hard triplet:

            d(r_a, r_p) = 1.4
            d(r_a, r_n) = 0.5
            m = 0.1
            L = max( 0, m+d(r_a, r_p)-d(r_a, r_n) ) = max(0, 0.1 + 1.4 - 0.5) = 1.0
        
        Semi triplet:

            d(r_a, r_p) = 0.45
            d(r_a, r_n) = 0.5
            m = 0.2
            L = max( 0, m+d(r_a, r_p)-d(r_a, r_n) ) = max(0, 0.2 + .45 - 0.5) = 0.15

        Easy triplet:

            d(r_a, r_p) = 0.4
            d(r_a, r_n) = 1.6
            m = 0.1
            L = max(m, d(r_a, r_p) - d(r_a, r_n)) = max(0, 0.1 + 0.4 - 1.6) = 0.0

    Batch all strategy - Select all the hard and semi-hard triplets and average the loss.
    A crucial point here is to not take into account the easy triplets (those with loss 0), 
    as averaging on them would make the overall loss very small.

    Args:
        mask (torch.Tensor): binary mask matrix of positive pairs (r_a, r_p)
        dist (torch.Tensor): distance matrix
        margin (float): margin for triplet loss

    Returns:
        torch.Tensor: [description]
    """

    p = dist[mask] - margin
    n = dist[torch.logical_not(mask)] - margin
    ap = torch.unsqueeze(p, 0)
    an = torch.unsqueeze(n, 1)
    triplet_loss = margin + ap - an
    triplet_loss = torch.maximum(torch.zeros_like(triplet_loss), triplet_loss)

    return triplet_loss


# @LOSSES.register_module()
class RankingLoss(nn.Module):

    reduction: str
    loss_weight: float
    margin: float
    mining_strategy: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
    dist_metric: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, 
        margin: float=0.05, 
        loss_weight: float=1.0, 
        mining_strategy: str='batch_all', 
        dist_metric: str='cosine',
        reduction: str='mean', 
        ):

        _strategies = dict(batch_all=triplet_batch_all)
        _dist_metrics = dict(cosine=CosineDistance())
        
        # preconditions
        assert reduction in ('none', 'mean', 'sum')
        assert isinstance(loss_weight, float)
        assert isinstance(margin, float)
        assert margin>0
        assert isinstance(mining_strategy, str)
        assert mining_strategy in _strategies
        assert isinstance(dist_metric, str)
        assert dist_metric in _dist_metrics

        self.margin = margin
        self.loss_weight = loss_weight
        self.mining_strategy = _strategies[mining_strategy]
        self.dist_metric = _dist_metrics[dist_metric]
        self.reduction = reduction

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = self.loss_weight * self.ranking_loss_fn(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor
        )

        return loss

    def _ranking_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
        ) -> torch.Tensor:
        r"""A warpper of cuda version `Ranking Loss`.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the number
                of classes.
            target (torch.Tensor): The learning label of the prediction.
        """
        gt_mask = _gt_mask(labels=target)
        distances = self.dist_metric(pred)
        triplet_loss = self.mining_strategy(gt_mask, distances)
    
        return triplet_loss

    def ranking_loss_fn(
        self,
        pred,
        target,
        weight=None,
        reduction='mean',
        avg_factor=None
        ):
        r"""A warpper of cuda version `Rank Loss`.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the number
                of classes.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        # Function.apply does not accept keyword arguments, so the decorator
        # "weighted_loss" is not applicable
        loss = self._ranking_loss(pred=pred, target=target)
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = weight.view(-1, 1)
                else:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim
        # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss


if __name__ == "__main__":
    
    import numpy as np

    latent_space = torch.FloatTensor([
        [1.0, 0.5, 0.3],
        [0.0, -0.6, 2.0],
        [1.1, 0.45, 0.25],
        [-5.0, 0.2 , -3.0]
    ])
    labels = torch.LongTensor(
        [1, 0, 1, 0]
    )

    rakn_loss = RankingLoss()
    loss = rakn_loss(pred=latent_space, target=labels)

    print(loss)

    