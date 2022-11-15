import pytest
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from MetricsReloaded.processes.mixed_measures_processes import MultiLabelLocSegPairwiseMeasure as MLIS
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import average_precision_score as aps
from MetricsReloaded.utility.utils import trapezoidal_integration
from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures


def test_auc():
    ref = np.asarray([0,0,0,1,1,1])
    pred_proba  = np.asarray([0.21,0.35,0.63, 0.92,0.32,0.79])
    ppm = ProbabilityPairwiseMeasures(pred_proba, ref)
    value_test = ppm.auroc()
    print(value_test)
    expected_auc = 0.78
    assert_allclose(value_test, expected_auc, atol=0.01)

def test_ap():
    ref = np.asarray([0,0,0,1,1,1])
    pred_proba = np.asarray([0.21, 0.35, 0.63, 0.92, 0.32, 0.79])
    ppm = ProbabilityPairwiseMeasures(pred_proba, ref)
    threshs = [0, 0.21, 0.32, 0.35, 0.63, 0.79, 0.92]
    recall = [1,1,0.66667,0.66667,0.66667,0.33,0]
    prec =[0.5, 0.6, 0.5, 0.66667, 1, 1, 1]
    expected_ap = trapezoidal_integration(np.asarray(recall)[::-1], np.asarray(prec)[::-1])
    print("From SK", prc(ref, pred_proba))

    expected_aps = aps(ref, pred_proba)
    value_test = ppm.average_precision()
    assert_allclose(value_test, expected_ap, atol=0.01)



def test_sensitivity_at_specificity():
    ref = np.concatenate([np.zeros([50]), np.ones([50])])
    pred = np.arange(0,1,0.01)
    ppm = ProbabilityPairwiseMeasures(pred, ref)
    value_sensspec = ppm.sensitivity_at_specificity()
    value_specsens = ppm.specificity_at_sensitivity()
    value_sensppv = ppm.sensitivity_at_ppv()
    value_ppvsens = ppm.ppv_at_sensitivity()
    expected_sensatspec = 1.0
    expected_specatsens = 1.0
    expected_sensatppv = 1.0
    expected_ppvatsens = 1.0
    assert_allclose(value_sensspec, expected_sensatspec, atol=0.01)
    assert_allclose(value_sensppv, expected_sensatppv, atol=0.01)
    assert_allclose(value_specsens, expected_specatsens, atol=0.01)
    assert_allclose(value_ppvsens, expected_ppvatsens, atol=0.01)

