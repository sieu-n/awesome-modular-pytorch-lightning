import torchvision.datasets as TD

try:
    from mmdet.datasets import build_dataset as build_dataset_mmdet
except ImportError:
    pass

try:
    from mmpose.datasets import build_dataset as build_dataset_mmpose
except ImportError:
    pass


def TorchvisionDataset(name, args):
    ds_builder = getattr(TD, name)
    return ds_builder(**args)


def MMDetectionDataset(cfg):
    print(f"Loading `{cfg['type']}` dataset from `mmdetection`.")
    return build_dataset_mmdet(cfg)


def MMPoseDataset(cfg):
    print(f"Loading `{cfg['type']}` dataset from `mmpose`.")
    return build_dataset_mmpose(cfg)
