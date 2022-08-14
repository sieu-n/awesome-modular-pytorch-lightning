import torch
from lightning.base import _BaseLightningTrainer


class PoseLiftingTrainer(_BaseLightningTrainer):
    """
    LightningModule for the paper:
        - A simple yet effective baseline for 3d human pose estimation
    where single 2d pose is given as input for making predictions about the 3d pose.
    """

    def init(self, model_cfg, training_cfg):
        # mixup and cutmix for classification
        self.get_decoded = training_cfg.get("get_decoded", True)
        self.normalization_mean = torch.tensor(self.const_cfg["normalization_mean"])
        self.normalization_std = torch.tensor(self.const_cfg["normalization_std"])

    def forward(self, x):
        reconstruction = self.backbone(x)
        return {
            "joints": reconstruction,
        }

    def _training_step(self, batch, batch_idx=0):
        """
        Something like:
        {
            "joint": tensor(batch_size, 17, 3),
            "joint_2d": tensor(batch_size, 17, 2),
            "meta": dict,
            "location": tensor(batch_size),
            "camera": Human36Camera,
            "idx": {
                "action_idx": list,
                "subaction_idx": list,
                "camera_idx": list,
                "frame_idx": list,
            }
        }
        """
        assert "joint_2d" in batch
        assert "joint" in batch
        assert "location" in batch
        assert "meta" in batch
        assert "idx" in batch
        assert "camera" in batch

        x, y = batch["joint_2d"], batch["joint"]

        pred = self(x)
        reconstruction = pred["joints"]

        loss = self.loss_fn(reconstruction, y)

        self.log("step/train_loss", loss)

        # for logging
        location = batch["location"]
        res = {
            "joints_gt_camera": y,
            "joints_2d": x,
            "reconstruction_camera": reconstruction,
            "loss": loss,
            "action_idx": batch["idx"]["action_idx"].cpu().numpy(),
        }
        if self.get_decoded:
            cameras = batch["camera"].data

            res["joints_gt_global"] = self.decode(y.cpu(), location.cpu(), cameras)
            res["reconstruction_global"] = self.decode(
                reconstruction.cpu().detach(), location.cpu(), cameras
            )

        return loss, res

    def evaluate(self, batch, stage=None):
        """
        Something like:
        {
            "joint": tensor(batch_size, 17, 3),
            "joint_2d": tensor(batch_size, 17, 2),
            "meta": dict,
            "location": tensor(batch_size),
            "camera": Human36Camera,
        }
        """
        assert "joint_2d" in batch
        assert "joint" in batch
        assert "location" in batch
        assert "meta" in batch
        assert "idx" in batch
        assert "camera" in batch

        x, y = batch["joint_2d"], batch["joint"]

        pred = self(x)
        reconstruction = pred["joints"]

        loss = self.loss_fn(reconstruction, y)

        # for logging
        camera = batch["camera"].data
        location = batch["location"]
        res = {
            "joints_gt_camera": y,
            "joints_gt_global": self.decode(y.cpu(), location.cpu(), camera),
            "joints_2d": x,
            "reconstruction_camera": reconstruction,
            "reconstruction_global": self.decode(
                reconstruction.cpu(), location.cpu(), camera
            ),
            "loss": loss,
            "action_idx": batch["idx"]["action_idx"].cpu().numpy(),
        }

        if "precomputed_joints_2d" in batch:
            reconstruction = self(batch["precomputed_joints_2d"])["joints"]
            res["pose_estimator_reconstruction_global"] = self.decode(
                reconstruction.cpu(), location.cpu(), camera
            )

        return res

    def _predict_step(self, batch, batch_idx=0):
        assert "images" in batch
        x = batch["images"]
        pred = self(x)
        return pred["logits"]

    def decode(self, joints, locations, cameras):
        """
        Parameters
        ----------
        joints: tensor
            bs x 17 x 3 shaped 3d joints denoting relative locations in camera
            coordinates.
        locations: tensor
            bs shaped locations denoting global hip joint location
        cameras: list[Human36Camera]
            bs number of cameras for each joint.
        Returns
        -------
        joints: tensor
            Joints in the global coordinate system.
        """
        # unnormalize
        batch_size = joints.size(0)
        ndim = self.normalization_mean.ndim

        normalization_mean = self.normalization_mean.repeat([batch_size] + [1] * ndim)
        normalization_std = self.normalization_std.repeat([batch_size] + [1] * ndim)

        joints = joints * normalization_std + normalization_mean

        # decenter
        joints = locations.tile(17, 1, 1).permute(1, 0, 2) + joints

        # to world coord
        joints = [
            cameras[idx].camera_to_world_coord(joint)
            for idx, joint in enumerate(joints)
        ]
        return torch.tensor(joints)


class VideoPoseSemisupTrainer(_BaseLightningTrainer):
    """
    LightningModule for semi-supervised training of the paper:
        - 3D human pose estimation in video with temporal convolutions and semi-supervised training.
    """

    def init(self, model_cfg, training_cfg):
        # mixup and cutmix for classification
        self.normalization_mean = torch.tensor(self.const_cfg["normalization_mean"])
        self.normalization_std = torch.tensor(self.const_cfg["normalization_std"])

    def forward(self, x):
        reconstruction = self.backbone(x)
        return {
            "joints": reconstruction,
        }

    def _training_step(self, batch, batch_idx=0):
        """
        Something like:
        {
            "joint": tensor(batch_size, 17, 3),
            "joint_2d": tensor(batch_size, 17, 2),
            "meta": dict,
            "location": tensor(batch_size),
            "camera": Human36Camera,
            "idx": {
                "action_idx": list,
                "subaction_idx": list,
                "camera_idx": list,
                "frame_idx": list,
            }
        }
        """
        assert "joint_2d" in batch
        assert "joint" in batch
        assert "location" in batch
        assert "meta" in batch
        assert "idx" in batch
        assert "camera" in batch

        x, y = batch["joint_2d"], batch["joint"]

        pred = self(x)
        reconstruction = pred["joints"]

        loss = self.loss_fn(reconstruction, y)

        self.log("step/train_loss", loss)

        # for logging
        camera = batch["camera"].data
        location = batch["location"]
        res = {
            "joints_gt_camera": y,
            "joints_gt_global": self.decode(y.cpu(), location.cpu(), camera),
            "joints_2d": x,
            "reconstruction_camera": reconstruction,
            "reconstruction_global": self.decode(
                reconstruction.cpu().detach(), location.cpu(), camera
            ),
            "loss": loss,
            "action_idx": batch["idx"]["action_idx"].cpu().numpy(),
        }
        return loss, res

    def evaluate(self, batch, stage=None):
        """
        Something like:
        {
            "joint": tensor(batch_size, 17, 3),
            "joint_2d": tensor(batch_size, 17, 2),
            "meta": dict,
            "location": tensor(batch_size),
            "camera": Human36Camera,
        }
        """
        assert "joint_2d" in batch
        assert "joint" in batch
        assert "location" in batch
        assert "meta" in batch
        assert "idx" in batch
        assert "camera" in batch

        x, y = batch["joint_2d"], batch["joint"]

        pred = self(x)
        reconstruction = pred["joints"]

        loss = self.loss_fn(reconstruction, y)

        # for logging
        camera = batch["camera"].data
        location = batch["location"]
        return {
            "joints_gt_camera": y,
            "joints_gt_global": self.decode(y.cpu(), location.cpu(), camera),
            "joints_2d": x,
            "reconstruction_camera": reconstruction,
            "reconstruction_global": self.decode(
                reconstruction.cpu(), location.cpu(), camera
            ),
            "loss": loss,
            "action_idx": batch["idx"]["action_idx"].cpu().numpy(),
        }

    def _predict_step(self, batch, batch_idx=0):
        assert "images" in batch
        x = batch["images"]
        pred = self(x)
        return pred["logits"]

    def decode(self, joints, locations, cameras):
        """
        joints: tensor
            bs x 17 x 3 shaped 3d joints denoting relative locations in camera
            coordinates.
        locations: tensor
            bs shaped locations denoting global hip joint location
        cameras: list[Human36Camera]
            bs number of cameras for each joint.
        Returns
        -------
        joints: tensor
            Joints in the global coordinate system.
        """
        # unnormalize
        batch_size = joints.size(0)
        DEVICE = joints.device
        ndim = self.normalization_mean.ndim

        normalization_mean = self.normalization_mean.repeat(
            [batch_size] + [1] * ndim
        ).to(DEVICE)
        normalization_std = self.normalization_std.repeat([batch_size] + [1] * ndim).to(
            DEVICE
        )

        joints = joints * normalization_std + normalization_mean

        # decenter
        joints = locations.tile(17, 1, 1).permute(1, 0, 2) + joints

        # to world coord
        joints = [
            cameras[idx].camera_to_world_coord(joint)
            for idx, joint in enumerate(joints)
        ]
        return torch.tensor(joints)
