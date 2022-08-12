from copy import deepcopy

import numpy as np


class Human36Camera:
    """
    Create camera class for transforming coordinates with regards to the camera
    for the Human3.6M 3D pose estimation dataset.

    Parameters
    ----------
    R: list[list]
        3x3 Camera rotation matrix
    f: float
        float(scalar) Camera focal length
    c: list
        2x1 Camera center
    T: list
        3x1 Camera translation parameters
    """

    def __init__(self, R, c, f, t):
        self.R = np.array(R)
        self.c = np.array(c)
        self.f = np.array(f)
        # t is given relative to initial point, unlike in original dataset.
        camPos = R @ (-np.array(t).reshape(-1, 1))[:, 0]
        self.t = np.expand_dims(camPos, 1)  # np.array(t)

    def world_to_camera_coord(self, P):
        """
        Convert points from world to camera coordinates
        Args
        P: Nx3 3d points in world coordinates(x, y, z order where z is height)
        Returns
        X_cam: Nx3 3d points in camera coordinates
        """
        assert len(P.shape) == 2
        assert P.shape[1] == 3

        X_cam = self.R.dot(P.T - self.t)  # rotate and translate

        return X_cam.T

    def camera_to_world_coord(self, P):
        """Inverse of world_to_camera_frame
        Args
            P: Nx3 points in camera coordinates
        Returns
            X_cam: Nx3 points in world coordinates
        """
        assert len(P.shape) == 2
        assert P.shape[1] == 3

        X_cam = self.R.T.dot(P.T) + self.t  # rotate and translate

        return X_cam.T

    def project_to_2D(self, P, is_world_coord=True):
        """
        Project points from 3d to 2d using camera parameters
        NOT including radial and tangential distortion
        https://github.com/una-dinosauria/3d-pose-baseline/blob/666080d86a96666d499300719053cc8af7ef51c8/src/data_utils.py#L253
        Args
          P: Nx3 points in world / camera coordinates
          is_world_coord: whether coordinates are in world / camera coordinate space.
          # k: 3x1 Camera radial distortion coefficients
          # p: 2x1 Camera tangential distortion coefficients
        Returns
          Proj: Nx2 points in pixel space
          D: 1xN depth of each point in camera space
          # radial: 1xN radial distortion per point
          # tan: 1xN tangential distortion per point
          r2: 1xN squared radius of the projected points before distortion
        """

        # P is a matrix of 3-dimensional points
        P = deepcopy(P)
        assert len(P.shape) == 2
        assert P.shape[1] == 3

        if is_world_coord:
            P = self.world_to_camera_coord(P)

        X = P.T
        XX = X[:2, :] / X[2, :]
        r2 = XX[0, :] ** 2 + XX[1, :] ** 2

        # don't consider camer focal coefficients for now.
        # N = P.shape[0]
        # radial = 1 + np.einsum( 'ij,ij->j', np.tile(self.k,(1, N)), np.array([r2, r2**2, r2**3]) )
        # tan = p[0]*XX[1,:] + p[1]*XX[0,:]

        # XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )
        Proj = (self.f.reshape(-1, 1) * XX) + self.c.reshape(-1, 1)
        Proj = Proj.T

        D = X[
            2,
        ]

        return Proj, D, r2
