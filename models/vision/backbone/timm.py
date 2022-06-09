import timm

from utils.models import drop_layers_after


def timm_feature_extractor(model_id, drop_after=None, reset_classifier=True, *args, **kwargs):
    """
    Load model(and pretrained-weights) implemented in `timm`.

    Model catalog can be found in: https://rwightman.github.io/pytorch-image-models/models

    Parameters
    ----------
    model_id: str
        Exact name of model to use. We look for `torchvision.models.{model_id}`.
    reset_classifier: bool
        Reset classifier automatically drops the classifier without the need to specify `drop_after` for most timm
        models.
    drop_after: str
        Remove all the layers including and after `drop_after` layer. For example, in the example below of `resnet50`,
        we want to set `drop_after` = "avgpool" to drop the final 2 layers and output feature map.
        ```
                ...
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
            )
            (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
            (fc): Linear(in_features=512, out_features=1000, bias=True)
        )
        ```
    Returns
    -------
    nn.Module
        feature_extractor network that can be used in multiple subtasks by plugging in different downstream heads.
    """
    # detach final classification head(make it feature extractor)
    # find model with same id & create model
    model = timm.create_model(model_id, num_classes=0, *args, **kwargs)
    if reset_classifier:
        model.reset_classifier(0, "")
    # detach final classification head(make it feature extractor)
    if drop_after:
        model = drop_layers_after(model, drop_after)

    return model
