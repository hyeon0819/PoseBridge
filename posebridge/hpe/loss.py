# ------------------------------------------------------------------------------
# Loss functions for RTMPose
# Based on mmpose official implementation
# KLDiscretLoss: Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    
    Modified from `the official implementation.
    <https://github.com/leeyegy/SimCC>`_.
    
    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_softmax (bool): Whether to use Softmax on labels. Default: False.
        label_beta (float): Temperature factor of Softmax on labels. Default: 10.0.
        use_target_weight (bool): Option to use weighted loss. Default: True.
        mask (list[int]): Index of masked keypoints. Default: None.
        mask_weight (float): Weight of masked keypoints. Default: 1.0.
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        label_softmax: bool = False,
        label_beta: float = 10.0,
        use_target_weight: bool = True,
        mask: Optional[list] = None,
        mask_weight: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')
    
    def criterion(self, dec_outs: Tensor, labels: Tensor) -> Tensor:
        """Criterion function.
        
        Args:
            dec_outs (Tensor): Predictions. Shape: [N, C].
            labels (Tensor): Target labels. Shape: [N, C].
        
        Returns:
            Tensor: Loss values. Shape: [N].
        """
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss
    
    def forward(
        self,
        pred_simcc: Tuple[Tensor, Tensor],
        gt_simcc: Tuple[Tensor, Tensor],
        target_weight: Tensor,
    ) -> Tensor:
        """Forward function.
        
        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis. Each has shape [N, K, L].
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
                Each has shape [N, K, L].
            target_weight (Tensor): Weights across different labels.
                Shape: [N, K] or [N, K, 1].
        
        Returns:
            Tensor: Loss value (scalar).
        """
        N, K, _ = pred_simcc[0].shape
        loss = 0
        
        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.
        
        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))
            
            t_loss = self.criterion(pred, target).mul(weight)
            
            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight
            
            loss = loss + t_loss.sum()
        
        return loss / K


def get_loss(cfg) -> KLDiscretLoss:
    """Build loss function from config.
    
    Args:
        cfg: Configuration object.
    
    Returns:
        KLDiscretLoss: Loss function.
    """
    loss = KLDiscretLoss(
        beta=cfg.LOSS.BETA,
        label_softmax=cfg.LOSS.LABEL_SOFTMAX,
        label_beta=cfg.LOSS.BETA,  # Same as beta in mmpose config
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
    )
    return loss
