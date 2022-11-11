from MetricsReloaded.metrics.calibration_measures import CalibrationMeasures
from numpy.testing import assert_allclose
import numpy as np

def test_expected_calibration_error():
    f40_pred = [0.22, 0.48, 0.49, 0.96, 0.55, 0.64, 0.78, 0.82, 0.34, 0.87]
    f40_ref = [0, 1, 0, 0, 1,1,1,1,1,0]
    ppm = CalibrationMeasures(f40_pred, f40_ref)
    ppm1 = CalibrationMeasures(f40_pred,f40_ref,dict_args={'bins_ece':2})
    value_test2 = ppm.expectation_calibration_error()
    value_test1 = ppm1.expectation_calibration_error()
    expected_ece1 = 0.11
    expected_ece2 = 0.36
    assert_allclose(value_test1, expected_ece1, atol=0.01)
    assert_allclose(value_test2, expected_ece2, atol=0.01)
    
def test_logarithmic_score():
    ref_ls = [1,0]
    pred_ls = [0.8, 0.6]
    ppm = CalibrationMeasures(np.asarray(pred_ls), np.asarray(ref_ls))
    value_test = ppm.logarithmic_score()
    expected_ls = -0.57
    assert_allclose(expected_ls, value_test, atol=0.01)
    

def test_brier_score():
    ref_bs = [1,0]
    pred_bs = [0.8, 0.6]
    ppm = CalibrationMeasures(np.asarray(pred_bs), np.asarray(ref_bs))
    value_test = ppm.brier_score()
    expected_bs = 0.2
    assert_allclose(expected_bs, value_test, atol=0.01)

def test_top_label_classification_error():
    ref_tce = [1, 0, 2, 1]
    pred_tce = [[0.1, 0.8, 0, 0.1],
                [0.6, 0.1, 0, 0.7],
                [0.3, 0.1, 1, 0.2 ]]
    pred_tce = np.asarray(pred_tce)
    ref_tce = np.asarray(ref_tce)
    expected_prob = [0.5, 0.25, 0.25, 0.5]
    best_prob = [0.6, 0.8, 1, 0.7]
    pred_class = [1, 0, 2, 1]
    expected_tce = 0.478
    cm = CalibrationMeasures(pred_tce, ref_tce)
    value_test = cm.top_label_classification_error()
    assert_allclose(value_test, expected_tce, atol=0.001)
    
