import torchmetrics
from torch import tensor


class MPJPE(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.add_state("dist", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, reconstructed_joints, gt_joints):
        self.total += len(reconstructed)

        dist_across = list(range(1, reconstructed.ndim + 1))
        self.dist += (reconstructed - gt_pose).pow(2).sum(dim=dist_across).sqrt().sum()

    def compute(self):
        return self.dist / self.total

    def reset(self):
        self.dist = 0
        self.total = 0
