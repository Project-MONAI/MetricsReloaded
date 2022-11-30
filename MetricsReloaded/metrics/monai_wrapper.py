"""Implementation for wrapping MetricsReloaded metrics as MONAI metrics.

See test/test_metrics/{test_binary_monai_metrics copy.py, test_categorical_monai_metrics.py}
for example use cases.

"""
try:
    from monai.metrics import CumulativeIterationMetric
except ImportError as e:
    raise ImportError("MONAI is not installed, please install MetricsReloaded with pip install .[monai]")
from monai.metrics.utils import (do_metric_reduction, is_binary_tensor)
from monai.utils import MetricReduction
import torch

from MetricsReloaded.metrics.pairwise_measures import (
    BinaryPairwiseMeasures,
    MultiClassPairwiseMeasures,
)


def torch2numpy(x):
    return x.cpu().detach().numpy()


def numpy2torch(x, dtype, device):
    return torch.as_tensor(x, dtype=dtype, device=device)


class Metric4Monai(CumulativeIterationMetric):
    """Allows for defining MetricsReloaded metrics as a CumulativeIterationMetric in MONAI
    """
    def __init__(
        self,
        metric_name,
        reduction=MetricReduction.MEAN,
        get_not_nans=False,
    ) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(self, reduction=None):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class BinaryMetric4Monai(Metric4Monai):
    """Wraps the binary pairwise metrics of MetricsReloaded
    """
    def __init__(
        self,
        metric_name,
        reduction=MetricReduction.MEAN,
        get_not_nans=False,
    ) -> None:
        super().__init__(metric_name=metric_name, reduction=reduction, get_not_nans=get_not_nans)

    def _compute_tensor(self, y_pred, y):
        """
        Args:
            y_pred: Prediction with dimensions (batch, channel, *spatial), where channel=1. The values should be binarized.
            y: Ground-truth with dimensions (batch, channel, *spatial), where channel=1. The values should be binarized.

        Raises:
            ValueError: when `y` or `y_pred` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
            ValueError: when second dimension ~= 1
        """
        # Sanity check
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        if y_pred.shape[1] != 1 or y.shape[1] != 1:
            raise ValueError(f"y_pred.shape[1]={y_pred.shape[1]} and y.shape[1]={y.shape[1]} should be one.")
        # Tensor parameters
        device = y_pred.device
        dtype = y_pred.dtype

        # To numpy array
        y_pred = torch2numpy(y_pred)
        y = torch2numpy(y)

        # Create binary pairwise metric object
        bpm = BinaryPairwiseMeasures(
            y_pred, y, axis=tuple(range(2, dims)), smooth_dr=1e-5,
        )

        # Is requested metric available?
        if self.metric_name not in bpm.metrics:
            raise ValueError(
                f"Unsupported metric: {self.metric_name}"
            )

        # Compute metric
        metric = bpm.metrics[self.metric_name]()

        # Return metric as numpy array
        return numpy2torch(metric, dtype, device)


class CategoricalMetric4Monai(Metric4Monai):
    """Wraps the categorical pairwise metrics of MetricsReloaded
    """
    def __init__(
        self,
        metric_name,
        reduction=MetricReduction.MEAN,
        get_not_nans=False,
    ) -> None:
        super().__init__(metric_name=metric_name, reduction=reduction, get_not_nans=get_not_nans)

    def _compute_tensor(self, y_pred, y):
        """
        Args:
            y_pred: Prediction with dimensions (batch, channel, *spatial). The values should be one-hot encoded and binarized.
            y: Ground-truth with dimensions (batch, channel, *spatial). The values should be one-hot encoded and binarized.

        Raises:
            ValueError: when `y` or `y_pred` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        # Sanity check
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # Tensor parameters
        device = y_pred.device
        dtype = y_pred.dtype
        num_classes = y_pred.shape[1]

        # Reshape for compatible dimension
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)
        y_pred = y_pred.permute((0, 2, 1))
        y = y.reshape(y.shape[0], y.shape[1], -1)
        y = y.permute((0, 2, 1))
        dims = y_pred.ndimension()

        # To numpy array
        y_pred = torch2numpy(y_pred)
        y = torch2numpy(y)

        # Create binary pairwise metric object
        bpm = MultiClassPairwiseMeasures(
            y_pred, y, axis=tuple(range(1, dims)), smooth_dr=1e-5,
            list_values=list(range(num_classes)), is_onehot=True,
        )

        # Is requested metric available?
        if self.metric_name not in bpm.metrics:
            raise ValueError(
                f"Unsupported metric: {self.metric_name}"
            )

        # Compute metric
        metric = bpm.metrics[self.metric_name]()

        # Put back channel dimension
        metric = metric[..., None]

        # Return metric as numpy array
        return numpy2torch(metric, dtype, device)
