from torch import nn
import torch.nn.functional as F


class FitNetCriterion(nn.Module):
    """
    Implement the paper:
        - FitNets: Hints for Thin Deep Nets

    """
    def __init__(self, channels, alpha, connector_cfg={}):
        pass

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
