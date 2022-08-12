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
        """
        Compute the distance between batch of 3d joint.

        Parameters
        ----------
        reconstructed_joints: Tensor
            batch_size x num_joints x 3
        gt_joints: Tensor
            batch_size x num_joints x 3
        """
        self.total += len(reconstructed_joints)

        self.dist += (
            (reconstructed_joints - gt_joints).pow(2).sum(dim=2).sqrt().mean()
        )

    def compute(self):
        return self.dist / self.total

    def reset(self):
        self.dist = 0
        self.total = 0
