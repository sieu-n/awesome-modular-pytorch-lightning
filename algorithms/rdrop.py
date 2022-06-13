# Implement utilities for implementing R-Drop:
# R-Drop: Regularized Dropout for Neural Networks
# code is based on official implementation: https://github.com/dropreg/R-Drop
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    # https://github.com/dropreg/R-Drop/blob/3d97565595747f3b3d9c4701cb2fb824a9139913/vit_src/models/modeling.py#L290
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # NOTE sum is for image classification
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
