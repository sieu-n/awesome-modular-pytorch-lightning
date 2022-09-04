# Camera is considered to be an instance of utils.camera.Human36Camera
from .. import _BaseTransform


class WorldToCameraCoord(_BaseTransform):
    def __call__(self, d):
        cam = d["camera"]
        d["joint"] = cam.world_to_camera_coord(d["joint"])
        return d


class CameraToWorldCoord(_BaseTransform):
    def __call__(self, d):
        cam = d["camera"]
        d["joint"] = cam.camera_to_world_coord(d["joint"])
        return d


class Create2DProjection(_BaseTransform):
    def __init__(self, is_world_coord=True, *args, **kwargs):
        super(Create2DProjection, self).__init__(*args, **kwargs)
        self.is_world_coord = is_world_coord

    def __call__(self, d):
        assert "joint_2d" not in d
        cam = d["camera"]
        d["joint_2d"], _, _ = cam.project_to_2D(
            d["joint"], is_world_coord=self.is_world_coord
        )
        return d


class Create2DProjectionTemporal(_BaseTransform):
    """
    Temporal version of Create2DProjection.
    """

    def __init__(self, is_world_coord=True, *args, **kwargs):
        super(Create2DProjectionTemporal, self).__init__(*args, **kwargs)
        self.is_world_coord = is_world_coord

    def __call__(self, d):
        assert "joint_2d" not in d
        cam = d["camera"]
        orig_shape = d["temporal_joints"].shape
        d["joint_2d"] = cam.project_to_2D(
            d["temporal_joints"].reshape(-1, 3), is_world_coord=self.is_world_coord
        )[0].reshape(orig_shape[:-1] + (2,))
        return d
