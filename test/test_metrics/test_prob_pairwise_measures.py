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
    

def test_expected_calibration_error():
    f40_pred = [0.22, 0.48, 0.49, 0.96, 0.55, 0.64, 0.78, 0.82, 0.34, 0.87]
    f40_ref = [0, 1, 0, 0, 1,1,1,1,1,0]
    ppm = ProbabilityPairwiseMeasures(f40_pred, f40_ref)
    ppm1 = ProbabilityPairwiseMeasures(f40_pred,f40_ref,dict_args={'bins_ece':2})
    value_test2 = ppm.expectation_calibration_error()
    value_test1 = ppm1.expectation_calibration_error()
    expected_ece1 = 0.11
    expected_ece2 = 0.36
    assert_allclose(value_test1, expected_ece1, atol=0.01)
    assert_allclose(value_test2, expected_ece2, atol=0.01)
    
def test_logarithmic_score():
    ref_ls = [1,0]
    pred_ls = [0.8, 0.6]
    ppm = ProbabilityPairwiseMeasures(np.asarray(pred_ls), np.asarray(ref_ls))
    value_test = ppm.logarithmic_score()
    expected_ls = -0.57
    assert_allclose(expected_ls, value_test, atol=0.01)
    

def test_brier_score():
    ref_bs = [1,0]
    pred_bs = [0.8, 0.6]
    ppm = ProbabilityPairwiseMeasures(np.asarray(pred_bs), np.asarray(ref_bs))
    value_test = ppm.brier_score()
    expected_bs = 0.2
    assert_allclose(expected_bs, value_test, atol=0.01)
    
