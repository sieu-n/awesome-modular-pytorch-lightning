import json
import os
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.camera import Human36Camera


class Human36AnnotationDataset(Dataset):
    def __init__(self, base_dir, subjects=[1, 5, 6, 7, 8]):
        """
        Base class for human36 annotations dataset. The `__init__` method collects
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
        self.base_dir = base_dir
        self.subjects = subjects

        self.cameras = {}
        self.data = {}
        self.sampler = []

        for idx, subject_id in enumerate(subjects):
            print(
                f"Loading data from subject `{subject_id}` ({idx + 1} / {len(subjects)})..."
            )
            subject_data = self.load_subject_data(subject_id)

            # build cameras
            self.cameras[subject_id] = {
                idx: Human36Camera(**subject_data["camera"][idx])
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

                key = (action_idx, subaction_idx, camera_idx, frame_idx)
                self.data[key] = {
                    "joint": np.array(
                        subject_data["joint"][action_idx][subaction_idx][frame_idx]
                    ),
                    "idx": {
                        "suject_idx": int(subject_id),
                        "action_idx": int(action_idx),
                        "subaction_idx": int(subaction_idx),
                        "camera_idx": int(camera_idx),
                        "frame_idx": int(frame_idx),
                    },
                    "meta": sample_meta,
                    "bbox": subject_data["data"]["annotations"][sample_idx],
                    "camera": self.cameras[subject_id][camera_idx],
                }
                self.sampler.append(key)

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

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        # {meta(from data), }
        key = self.sampler[idx]
        return deepcopy(self.data[key])


class Human36AnnotationDatasetBase(Dataset):
    """
    Base class for human36 annotations dataset. The `__init__` method collects
    and caches the annotation data of every actor. `self.data` is a dictionary
    that contains the following attributes:
        - joint: 17 x 3 array of 3d joints
        - meta: metadata of frame
        - bbox: Ground-truth bounding boxes provided in the annotation file
        - camera: `Human36Camera` instance of that frame

    Parameters
    ----------
    base_dir: str
        directory to where dataset is saved
    subjects: list[int]
        Subjects to use while creating dataset. The common preocedure is to use
        subjects [1, 5, 6, 7, 8] for training and [9, 11] for validation.
    """

    def __init__(self, base_dir, subjects):
        self.base_dir = base_dir
        self.subjects = subjects

        self.cameras = {}

        self.joint_data = {}
        self.data = {}
        self.sampler = []
        for idx, subject_id in enumerate(subjects):
            print(
                f"Loading data from subject `{subject_id}` ({idx + 1} / {len(subjects)})..."
            )
            subject_data = self.load_subject_data(subject_id)
            self.data[subject_id] = {}
            # build cameras
            self.cameras[subject_id] = {
                idx: Human36Camera(**camera[idx])
                for idx in subject_data["camera"].keys()
            }

            _joint_data = {}
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
                frame_idx = int(sample_meta["frame_idx"])
                camera_idx = str(sample_meta["cam_idx"])

                _data = {
                    "meta": sample_meta,
                    "bbox": subject_data["data"]["annotations"][sample_idx],
                    # "camera": self.cameras[subject_id][camera_idx],
                }
                CLIP_IDX = f"{subject_id}_{action_idx}_{subaction_idx}_{camera_idx}"
                if CLIP_IDX not in self.data:
                    self.data[CLIP_IDX] = {}
                    print(f"Creating: {CLIP_IDX}")
                self.data[CLIP_IDX][frame_idx] = _data

                if CLIP_IDX not in _joint_data:
                    _joint_data[CLIP_IDX] = {}
                _joint_data[CLIP_IDX][frame_idx] = np.array(
                    subject_data["joint"][action_idx][subaction_idx][str(frame_idx)]
                )
                self.sampler.append(
                    {
                        "key": CLIP_IDX,
                        "subject": subject_id,
                        "action": action_idx,
                        "subaction": subaction_idx,
                        "camera": camera_idx,
                        "frame": frame_idx,
                    }
                )
        # organize joints into list
        for CLIP_IDX in _joint_data:
            d = _joint_data[CLIP_IDX]
            frames = [d[frame_idx] for frame_idx in range(len(d))]
            d = np.array(frames)
            self.joint_data[CLIP_IDX] = d
            print(f"Creating: {CLIP_IDX}", "  d.shape:", d.shape)

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

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        # {meta(from data), }
        sample = self.sampler[idx]
        frame_idx = sample["frame"]
        _data = self.data[sample["key"]]

        joints = deepcopy(self.joint_data[sample["key"]])

        sample_width = cfg["receptive_field"] // 2
        sample_left = max(frame_idx - sample_width, 0)
        sample_right = min(frame_idx + sample_width + 1, len(_data) - 1)

        joints = joints[sample_left:sample_right]
        if frame_idx - sample_width < 0:
            pad_len = sample_width - frame_idx
            padded = np.tile(joints[0], (pad_len, 1, 1))
            joints = np.concatenate((padded, joints), axis=0)

        if frame_idx + sample_width + 1 >= (len(_data) - 1):
            pad_len = frame_idx + sample_width + 1 - (len(_data) - 1)
            padded = np.tile(joints[-1], (pad_len, 1, 1))
            joints = np.concatenate((joints, padded), axis=0)

        return {
            "joint": self.joint_data[sample["key"]][frame_idx],
            "joints": joints,
            "meta": _data[frame_idx]["meta"],
            "bbox": _data[frame_idx]["bbox"],
            "camera": self.cameras[sample["subject"]][sample["camera"]],
        }
