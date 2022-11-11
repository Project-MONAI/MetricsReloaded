import pytest
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from MetricsReloaded.processes.mixed_measures_processes import MultiLabelLocSegPairwiseMeasure as MLIS
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures


def test_auc():
    ref = np.asarray([0,0,0,1,1,1])
    pred_proba  = np.asarray([0.21,0.35,0.63, 0.92,0.32,0.79])
    ppm = ProbabilityPairwiseMeasures(pred_proba, ref)
    value_test = ppm.auroc()
    print(value_test)
    expected_auc = 0.78
    assert_allclose(value_test, expected_auc, atol=0.01)
    

