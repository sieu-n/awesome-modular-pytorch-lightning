# .py config files from `mmdetection` is converted into `yaml`

Files here are show the original configurations from `mmdetection` before adaptation for reference. They can't be used for training.

These files were originally created using `tools/mmcv_config_to_yaml.py` which reads the `.py` config file and converts the [Config](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) class of `mmcv` into native Python dictionary. In addition, as tuples can't be saved into a `yaml` file, they are converted into lists. This processed dictionary is then saved into a `yaml` file.

For example, `faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.yaml` was created using:
```shell
python tools/mmcv_config_to_yaml.py -c {PATH TO MMDetection}/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py 
```
