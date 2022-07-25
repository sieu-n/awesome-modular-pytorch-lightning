# Torchmetrics can be integrated

Please have a read at the `torchmetrics` catalog to check the list of readily available metrics.
- <https://torchmetrics.readthedocs.io/en/stable/all-metrics.html>

Custom metrics can be implemented using the following guide. The metrics defined under `training.metrics` are
automatically applied during training.
- <https://torchmetrics.readthedocs.io/en/stable/pages/implement.html>

## Keywords

Examples:
```yaml
training:
  metrics:
    class-avg-accuracy:
      when: "trn,val,test"
      name: TorchMetric
      args:
        name: "Accuracy"
        args:
          average: "macro"
          num_classes: "{const.num_classes}"
      update:
        preds: "logits"
        target: "y"
```

Note the following keywords that can be provided under the `training.metrics`:

- `when`
- `name`
- `file`
- `args`
- `update`
