from MetricsReloaded.metrics.monai_wrapper import BinaryMetric4Monai
from monai.utils import set_determinism
import torch

set_determinism(seed=0)

n = 2
c = 1
x = 32
y = 32
z = 32

y_pred = (torch.rand(n, c, x, y, z) > 0.5).float()
y = (torch.rand(n, c, x, y, z) > 0.5).float()

metric_names = (
    "False Positives",
    "False Negatives",
    "True Positives",
    "True Negatives",
    "Youden Index",
    "Sensitivity",
    "Specificity",
    "Balanced Accuracy",
    "Accuracy",
    "False Positive Rate",
    "Normalised Expected Cost",
    "Matthews Correlation Coefficient",
    "Cohens Kappa",
    "Positive Likelihood Ratio",
    "Prediction Overlaps Reference",
    "Positive Predictive Value",
    "Recall",
    "FBeta",
    "Net Benefit Treated",
    "Negative Predictive Values",
    "Dice Score",
    "False Positives Per Image",
    "Intersection Over Reference",
    "Intersection Over Union",
    "Volume Difference",
    "Topology Precision",
    "Topology Sensitivity",
    "Centreline Dice Score",
    "Boundary IoU",
    "Normalised Surface Distance",
    "Average Symmetric Surface Distance",
    "Mean Average Surfance Distance",
    "Hausdorff Distance",
    "xTh Percentile Hausdorff Distance",
)

for metric_name in metric_names:
    print("=" * 32)
    print(metric_name)
    metric = BinaryMetric4Monai(metric_name=metric_name)
    value = metric(y_pred=y_pred, y=y)
    print(value)
    metric(y_pred=y_pred, y=y)
    value = metric.aggregate().item()
    print(value)
    metric.reset()