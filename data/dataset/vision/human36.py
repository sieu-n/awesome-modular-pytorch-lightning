import json
import os
from copy import deepcopy

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.camera import Human36Camera

ACTION_NAMES = (
    "Directions",
    "Discussion",
    "Eating",
    "Greeting",
    "Phoning",
    "Posing",
    "Purchases",
    "Sitting",
    "SittingDown",
    "Smoking",
    "Photo",
    "Waiting",
    "Walking",
    "WalkDog",
    "WalkTogether",
)


class PrecomputedJointDataset:
    def __init__(
        self,
        data_dir: str,
        subjects: list = [1, 5, 6, 7, 8],
        format: str = "videopose3d",
    ):
        self.data_dir = data_dir
        self.subjects = subjects
        print("Loading precomputed joints from %s" % self.data_dir)

        if format == "videopose3d":
            self.data, _ = self.load_videopose3d(data_dir)

    def __getitem__(self, key):
        return self.data[key]

    def load_videopose3d(self, data_dir: str):
        keypoints = np.load(data_dir, allow_pickle=True)

        keypoints_metadata = keypoints["metadata"].item()
        keypoints = keypoints["positions_2d"].item()

        data = {}

        for subject_name in tqdm(keypoints.keys()):
            subject_id = int(subject_name[1:])
            if subject_id not in self.subjects:
                continue

            for action_key in list(keypoints[subject_name].keys()):
                key = action_key.split(" ")
                if len(key) == 2:
                    action_name, subaction_idx = key
                    joint_key = (
                        subject_id,
                        ACTION_NAMES.index(action_name),
                        int(subaction_idx),
                    )
                elif len(key) == 1:
                    joint_key = (
                        subject_id,
                        ACTION_NAMES.index(action_key),
                        2,
                    )
                else:
                    raise ValueError(f"Invalid action name: {action_name}")

                data[joint_key] = keypoints[subject_name].pop(action_key)

        return data, keypoints_metadata


class Human36AnnotationDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        subjects: list = [1, 5, 6, 7, 8],
        cameras: list = None,
        reorder_joints: bool = True,
        precomputed_joint_dir: str = None,
        precomputed_joint_format: str = "videopose3d",
        image_dir: str = None,
    ):
        """
        Base class for human36 annotations dataset. The `__init__` method collects
        and caches the annotation data of every actor. `self.data` is a dictionary
        that contains the following attributes:
            - meta: metadata of frame
            - bbox: Ground-truth bounding boxes provided in the annotation file
            - camera: `utils.camera.Human36Camera` instance of that frame
        `self.data` can be parsed by specifying the action, subaction, camera, and frame
        index. The 3d joint coordinates are saved in a separate dictionary `joint_data`
        which is indexed by the same keys.
        This dataset can be used to implement single-frame 2d -> 3d lifting algorithms.

        Parameters
        ----------
        base_dir: str
            directory to where dataset is saved
        subjects: list[int]
            Subjects to use while creating dataset. The common preocedure is to use
            subjects [1, 5, 6, 7, 8] for training and [9, 11] for validation.
        """
        self.base_dir = base_dir
        self.subjects = subjects
        self.cameras_allowed = cameras

        self.cameras = {}
        self.joint_data = {}
        self.data = {}
        self.sampler = []
        self.image_dir = image_dir

        self.get_precomputed_joints = precomputed_joint_dir is not None
        if self.get_precomputed_joints:
            self.precomputed_joints = PrecomputedJointDataset(
                precomputed_joint_dir,
                subjects=subjects,
                format=precomputed_joint_format,
            )

        for idx, subject_id in enumerate(subjects):
            print(
                f"Loading data from subject `{subject_id}` ({idx + 1} / {len(subjects)})..."
            )
            subject_data = self.load_subject_data(subject_id)

            # build cameras
            self.cameras[subject_id] = {
                int(idx) - 1: Human36Camera(**subject_data["camera"][idx])
                for idx in subject_data["camera"].keys()
            }

            # load data
            for sample_idx, sample_meta in tqdm(
                enumerate(subject_data["data"]["images"])
            ):
                assert (
                    subject_data["data"]["annotations"][sample_idx]["id"]
                    == subject_data["data"]["images"][sample_idx]["id"]
                )
                action_idx = str(sample_meta["action_idx"])
                subaction_idx = str(sample_meta["subaction_idx"])
                camera_idx = str(sample_meta["cam_idx"])
                frame_idx = str(sample_meta["frame_idx"])

                # skip
                if (self.cameras_allowed is not None) and (
                    camera_idx not in self.cameras_allowed
                ):
                    continue

                # save joint data
                key = (
                    int(subject_id),
                    int(action_idx) - 2,
                    int(subaction_idx),
                    int(camera_idx) - 1,
                    int(frame_idx),
                )
                self.joint_data[key] = np.array(
                    subject_data["joint"][action_idx][subaction_idx][frame_idx]
                )
                # save other data
                self.data[key] = {
                    "idx": {  # values should be `int` type to be collated properly.
                        "suject_idx": int(subject_id),
                        "action_idx": int(action_idx) - 2,
                        "subaction_idx": int(subaction_idx),
                        "camera_idx": int(camera_idx) - 1,
                        "frame_idx": int(frame_idx),
                    },
                    "meta": sample_meta,
                    "bbox": subject_data["data"]["annotations"][sample_idx],
                }
                self.sampler.append(key)
        self.is_reorder_joints = reorder_joints
        if reorder_joints:
            self.reorder_joints(self.joint_data)

    def reorder_joints(self, joint_data):
        """
        Reorder joints so data of each scene is a numpy array of shape (# frames, 17, 3)
        Works in an inplace fashion.
        """
        # organize based on `scene_key` = (action_idx, subaction_idx)
        print(
            "Organizing frames based on `scene_key`(subject_id, action_idx, subaction_idx)"
        )
        _joint_data = {}
        for subject_id, action_idx, subaction_idx, camera_idx, frame_idx in tqdm(
            list(joint_data.keys())
        ):
            scene_key = (subject_id, action_idx, subaction_idx)
            if scene_key not in _joint_data:
                _joint_data[scene_key] = {}
            _joint_data[scene_key][frame_idx] = joint_data.pop(
                (subject_id, action_idx, subaction_idx, camera_idx, frame_idx)
            )
        assert (
            len(joint_data) == 0
        ), f"Expected dict to be empty, but contains: {joint_data}"
        # organize joints into numpy arrays
        print("Grouping dictionaries into numpy array")
        for scene_key in tqdm(_joint_data):
            d = _joint_data[scene_key]
            joint_data[scene_key] = np.array(
                [d[frame_idx] for frame_idx in range(len(d))]
            )

    def load_subject_data(self, subject_id):
        def load_json_data(subject_id, data_type):
            data_path = os.path.join(
                self.base_dir,
                f"annotations/Human36M_subject{subject_id}_{data_type}.json",
            )
            return json.load(open(data_path))

        return {
            "data": load_json_data(subject_id, "data"),
            "camera": load_json_data(subject_id, "camera"),
            "joint": load_json_data(subject_id, "joint_3d"),
        }

    def get_image(self, idx=None):
        return Image.open(
            os.path.join(
                self.image_dir, self.data[self.sampler[idx]]["meta"]["file_path"]
            )
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        # {meta(from data), }
        key = self.sampler[idx]
        subject_id, action_idx, subaction_idx, camera_idx, frame_idx = key

        if self.is_reorder_joints:
            joint_key = (subject_id, action_idx, subaction_idx)
        else:
            joint_key = key

        # organize results
        res = deepcopy(self.data[key])
        res["joint"] = deepcopy(self.joint_data[joint_key][frame_idx])
        res["camera"] = self.cameras[subject_id][camera_idx]
        if self.image_dir is not None:
            res["images"] = self.get_image(idx)
        if self.get_precomputed_joints:
            res["precomputed_joints_2d"] = self.precomputed_joints[joint_key][
                camera_idx
            ][frame_idx]
        return res


class Human36AnnotationTemporalDataset(Human36AnnotationDataset):
    def __init__(self, receptive_field: int = 243, *args, **kwargs):
        """
        Human36 temporal annotations dataset. The `__init__` method collects
        and caches the annotation data of every actor. `self.data` is a dictionary
        that contains the following attributes:
            - joint: 17 x 3 array of 3d joints
            - meta: metadata of frame
            - bbox: Ground-truth bounding boxes provided in the annotation file
            - camera: `utils.camera.Human36Camera` instance of that frame
        `self.data` can be parsed by specifying the action, subaction, camera, and frame
        index. This dataset can be used to implement single-frame 2d -> 3d lifting algorithms.

        Parameters
        ----------
        base_dir: str
            directory to where dataset is saved
        subjects: list[int]
            Subjects to use while creating dataset. The common preocedure is to use
            subjects [1, 5, 6, 7, 8] for training and [9, 11] for validation.
        """
        super().__init__(*args, **kwargs)
        self.receptive_field = receptive_field

    def __getitem__(self, idx):
        subject_id, action_idx, subaction_idx, camera_idx, frame_idx = self.sampler[idx]

        key = (subject_id, action_idx, subaction_idx, camera_idx, frame_idx)
        joint_key = (subject_id, action_idx, subaction_idx)
        num_frames = len(self.joint_data[joint_key])

        sample_width = self.receptive_field // 2
        sample_left = max(frame_idx - sample_width, 0)
        sample_right = min(frame_idx + sample_width + 1, num_frames - 1)

        joints = deepcopy(self.joint_data[joint_key][sample_left:sample_right])
        if frame_idx - sample_width < 0:
            # pad left
            pad_len = sample_width - frame_idx
            padded = np.tile(joints[0], (pad_len, 1, 1))
            joints = np.concatenate((padded, joints), axis=0)

        if frame_idx + sample_width + 1 >= num_frames - 1:
            # pad right
            pad_len = frame_idx + sample_width + 1 - (num_frames - 1)
            padded = np.tile(joints[-1], (pad_len, 1, 1))
            joints = np.concatenate((joints, padded), axis=0)

        # organize results
        res = deepcopy(self.data[key])
        assert self.receptive_field == len(joints)
        res["temporal_joints"] = joints
        res["joint"] = joints[sample_width]  # sample_width = self.receptive_field // 2
        res["camera"] = self.cameras[subject_id][camera_idx]
        if self.get_precomputed_joints:
            res["precomputed_joints_2d"] = self.precomputed_joints[joint_key][
                camera_idx
            ][frame_idx]
        return res
