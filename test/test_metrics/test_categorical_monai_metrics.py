from MetricsReloaded.metrics.monai_wrapper import CategoricalMetric4Monai
from monai.utils import set_determinism
from monai.networks.utils import one_hot
import torch

set_determinism(seed=0)

n = 2
c = 1
x = 32
y = 32
z = 32
num_classes = 3

y_pred = one_hot((num_classes*torch.rand(n, c, x, y, z)).round(), num_classes=num_classes + 1)
y = one_hot((num_classes*torch.rand(n, c, x, y, z)).round(), num_classes=num_classes + 1)

metric_names = (
    "Balanced Accuracy",
    "Weighted Cohens Kappa",
    "Matthews Correlation Coefficient",
    "Expected Cost",
    "Normalised Expected Cost",
)

for metric_name in metric_names:
    print("=" * 32)
    print(metric_name)
    metric = CategoricalMetric4Monai(metric_name=metric_name)
    value = metric(y_pred=y_pred, y=y)
    print(value)
    metric(y_pred=y_pred, y=y)
    value = metric.aggregate().item()
    print(value)
    metric.reset()