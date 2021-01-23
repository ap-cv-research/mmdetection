from typing import Callable
import torch
import torch.nn as nn
# from ..builder import LOSSES
# from .utils import weight_reduce_loss


__all__ = ('RankingLoss',)


def _get_triplet_mask(labels: torch.Tensor) -> torch.BoolTensor:
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels (torch.Tensor): `Tensor` with shape [batch_size]

    Returns:
        torch.BoolTensor: `Tensor` with shape [batch_size]
    """

    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels: torch.Tensor, device: torch.device) -> torch.BoolTensor:
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels (torch.Tensor): `Tensor` with shape [batch_size]
        device (torch.device): `Tensor` with shape [batch_size, batch_size]

    Returns:
        torch.BoolTensor: [description]
    """
    
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels: torch.Tensor) -> torch.BoolTensor:
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels (torch.Tensor): `Tensor` with shape [batch_size]

    Returns:
        torch.BoolTensor: `Tensor` with shape [batch_size, batch_size]
    """
    
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


class CosineDistance(object):

    _s: Callable

    def __init__(self, dim=2, eps=1e-6):

        self._s = nn.CosineSimilarity(dim=dim, eps=eps)

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:

        # preconditions
        assert isinstance(embeddings, torch.Tensor)
        assert len(embeddings.shape) == 2
        
        c = embeddings.shape[-1]
        t = embeddings.view(-1, 1, c)
        similarity = self._s(t, torch.transpose(t, dim0=1, dim1=0))
        
        return 0.5 - 0.5 * similarity


class L2Norm(object):

    squared: bool

    def __init__(self, squared: bool=False):
        
        assert isinstance(squared, bool)
        self.squared = squared

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        
        # preconditions
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.shape) == 2

        dot_product = torch.matmul(embeddings, embeddings.t())
        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not self.squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 -mask) * torch.sqrt(distances)

        return distances


def batch_hard_triplet_loss(labels: torch.Tensor, pairwise_dist: torch.Tensor, margin: float, device='cpu'):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels (torch.Tensor): labels of the batch, of size (batch_size,)
        pairwise_dist (torch.Tensor): pairwise distance matrix tensor of shape (batch_size, batch_size)
        margin (float): margin for triplet loss
        device (str, optional): [description]. Defaults to 'cpu'.

    Returns:
        [type]: scalar tensor containing the triplet loss
    """
    
    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl[tl < 0] = 0
    triplet_loss = tl.mean()

    return triplet_loss


def batch_all_triplet_loss(labels: torch.Tensor, pairwise_dist: torch.Tensor, margin: float, device='cpu'):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = triplet_loss.clamp_min(0.0) # triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss.gt(1e-16)
    num_positive_triplets = valid_triplets.size(0)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum(dim=(1,2)) / valid_triplets.sum(dim=(1,2)).clamp_min(1)

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
        mining_strategy: str='all', 
        dist_metric: str='cosine',
        reduction: str='mean', 
        ):

        _strategies = dict(all=batch_all_triplet_loss, hard=batch_hard_triplet_loss)
        _dist_metrics = dict(cosine=CosineDistance(), l2norm=L2Norm())
        
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
        distances = self.dist_metric(pred)
        triplet_loss = self.mining_strategy(target, distances, self.margin, target.device)
    
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
        [1.1, 0.45, 0.25]
    ])
    easy_labels = torch.LongTensor(
        [1, 0, 1]
    )

    hard_labels = torch.LongTensor(
        [1, 1, 0]
    )

    rakn_loss = RankingLoss()
    loss = rakn_loss.forward(pred=latent_space, target=hard_labels)
    loss = rakn_loss.forward(pred=latent_space, target=easy_labels)

    print(loss)

    