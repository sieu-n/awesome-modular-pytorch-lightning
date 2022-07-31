# Camera is considered to be an instance of utils.camera.Human36Camera
from ..base import _BaseTransform


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
    def __call__(self, d):
        assert "joint_2d" not in d
        cam = d["camera"]
        d["joint_2d"], _, _ = cam.project_to_2D(d["joint"])
        return d
