import torch.nn.functional as F
from torch import nn


class DistillLogits(nn.Module):
    """
    Base class for criterions that implement logit-based knowledge distillation.
    """

    def compute_loss(self, y_s, y_t):
        raise NotImplementedError()

    def forward(self, s_hook, t_hook):
        """
        Expect `s_hook` to have only one element.
        Parameters
        ----------
        s_hook: dict
            dictionary of hooks extracted from student model containing the logit.
        t_hook: dict
            dictionary of hooks extracted from teacher model containing the logit.
        Returns
        -------
        loss: Tensor
        res: None
            To return metrics for logging, override `forward` or extend the
            implementation after calling `super().forward(s_hook, t_hook)` inside
            the subclass.
        """
        y_s, y_t = self.get_logits(s_hook, t_hook)
        loss = self.compute_loss(y_s, y_t)
        return loss, {"teacher_logits": y_t}

    def get_logits(self, s_hook, t_hook):
        assert (
            isinstance(s_hook, dict) and len(s_hook) == 1
        ), f"Invalid value \
            were passed to criterion from student hook, got {s_hook} of type {type(s_hook)}"
        assert (
            isinstance(t_hook, dict) and len(t_hook) == 1
        ), f"Invalid value \
            were passed to criterion from teacher hook, got {t_hook} of type {type(t_hook)}"

        return next(iter(s_hook.values())), next(iter(t_hook.values()))


class LogitKLCriterion(DistillLogits):
    """
    Implement the default knowledge distillation algorithm from the paper:
        Distilling the Knowledge in a Neural Network, NIPS-W 2014
    Code from:
        https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/KD.py
    Parameters
    ----------
    T: float
        temperature term that is applied to soften the logits of the teacher and student.
    """

    def __init__(self, alpha: float, T: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.T = T

    def compute_loss(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss * self.alpha


class LogitMSECriterion(DistillLogits):
    """
    Implement the MSE-variant of knowledge distillation algorithm from the paper:
        Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation
    Code from:
        https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/KD.py
    Parameters
    ----------
    T: float
        temperature term that is applied to soften the logits of the teacher and student.
    """

    def __init__(self, alpha: float, T: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.T = T

    def compute_loss(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.mse_loss(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss * self.alpha
