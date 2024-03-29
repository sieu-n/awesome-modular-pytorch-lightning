import copy

from .. import _BaseTransform


class CenterAroundJoint(_BaseTransform):
    def __init__(self, center_joint_idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_joint_idx = center_joint_idx

    def __call__(self, d):
        d["joint"], d["location"] = self._center_around_joint(
            d["joint"], self.center_joint_idx
        )
        return d

    def _center_around_joint(self, P, center_joint_idx=0):
        """Center 3d points around root joint(hip)

        Parameters
        ----------
        P: np.array(17, n)
            pose with 3d or 2d data

        Returns
        -------
            P_relative: 3d data centred around root (center hip) joint
            root_position: original 3d position
        """
        root_position = copy.deepcopy(P[center_joint_idx])

        # Remove the root from the 3d position
        P_relative = P - P[..., [center_joint_idx], :]
        return P_relative, root_position
