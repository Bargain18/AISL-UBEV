from models.model import Model
from tools.loss import *
import torch.nn as nn


class Baseline(Model):
    def __init__(self, *args, **kwargs):
        super(Baseline, self).__init__(*args, **kwargs)

        self.m_in = -23.0
        self.m_out = -5.0

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)

    def loss(self, logits, target, reduction='mean'):
        # ce = ce_loss(logits, target, weights=self.weights)
        # focal = focal_loss(logits, target, weights=self.weights, n=self.gamma)
        # return (ce + focal).mean()
        if self.loss_type == 'ce':
            A = bce_loss(logits, target, weights=self.weights)
        elif self.loss_type == 'focal':
            A = focal_loss(logits, target, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A