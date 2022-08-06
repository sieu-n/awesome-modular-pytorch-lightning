import torch
import torchmetrics


class MPJPE(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.add_state(
            "dist", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, reconstructed_joints, gt_joints):
        self.total += len(reconstructed_joints)

        dist_across = list(range(1, reconstructed_joints.ndim))
        self.dist += (
            (reconstructed_joints - gt_joints).pow(2).sum(dim=dist_across).sqrt().sum()
        )

    def compute(self):
        return self.dist / self.total

    def reset(self):
        self.dist = 0
        self.total = 0
