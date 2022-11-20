from MetricsReloaded.metrics.calibration_measures import CalibrationMeasures
from numpy.testing import assert_allclose
import numpy as np
from scipy.special import gamma


def test_expected_calibration_error():
    f40_pred = [0.22, 0.48, 0.49, 0.96, 0.55, 0.64, 0.78, 0.82, 0.34, 0.87]
    f40_ref = [0, 1, 0, 0, 1, 1, 1, 1, 1, 0]
    ppm = CalibrationMeasures(f40_pred, f40_ref)
    ppm1 = CalibrationMeasures(f40_pred, f40_ref, dict_args={"bins_ece": 2})
    value_test2 = ppm.expectation_calibration_error()
    value_test1 = ppm1.expectation_calibration_error()
    expected_ece1 = 0.11
    expected_ece2 = 0.36
    assert_allclose(value_test1, expected_ece1, atol=0.01)
    assert_allclose(value_test2, expected_ece2, atol=0.01)


def test_logarithmic_score():
    ref_ls = [1, 0]
    pred_ls = [0.8, 0.6]
    ppm = CalibrationMeasures(np.asarray(pred_ls), np.asarray(ref_ls))
    value_test = ppm.logarithmic_score()
    expected_ls = -0.57
    assert_allclose(expected_ls, value_test, atol=0.01)


def test_brier_score():
    ref_bs = [1, 0]
    pred_bs = [0.8, 0.6]
    ppm = CalibrationMeasures(np.asarray(pred_bs), np.asarray(ref_bs))
    value_test = ppm.brier_score()
    expected_bs = 0.2
    assert_allclose(expected_bs, value_test, atol=0.01)


def test_top_label_classification_error():
    ref_tce = [1, 0, 2, 1]
    pred_tce = [[0.1, 0.8, 0, 0.1], [0.6, 0.1, 0, 0.7], [0.3, 0.1, 1, 0.2]]
    pred_tce = np.asarray(pred_tce).T
    ref_tce = np.asarray(ref_tce)
    expected_prob = [0.5, 0.25, 0.25, 0.5]
    best_prob = [0.6, 0.8, 1, 0.7]
    pred_class = [1, 0, 2, 1]
    expected_tce = 0.478
    cm = CalibrationMeasures(pred_tce, ref_tce)
    value_test = cm.top_label_classification_error()
    assert_allclose(value_test, expected_tce, atol=0.001)


def test_negative_log_likelihood():
    ref_nll = [1, 0, 2, 1]
    pred_nll = [[0.1, 0.8, 0.05, 0.1], [0.6, 0.1, 0, 0.7], [0.3, 0.1, 0.95, 0.2]]
    ref_nll = np.asarray(ref_nll)
    pred_nll = np.asarray(pred_nll).T
    expected_nll = -1 * (np.log(0.8) + np.log(0.6) + np.log(0.7) + np.log(0.95))
    cm = CalibrationMeasures(pred_nll, ref_nll)
    value_test = cm.negative_log_likelihood()
    assert_allclose(value_test, expected_nll)


def test_class_wise_expectation_calibration_error():
    ref_cwece = [1, 0, 2, 1]
    pred_cwece = [[0.1, 0.8, 0, 0.1], [0.6, 0.1, 0, 0.7], [0.3, 0.1, 1, 0.2]]
    # 0.06 * 3
    # 0.2 * 1
    # 0.05 * 2
    # 0.35 * 2
    # 0.2 * 3
    # 0 * 1
    ref_cwece = np.asarray(ref_cwece)
    pred_cwece = np.asarray(pred_cwece).T
    dict_args = {"bins_ece": 2}
    cm = CalibrationMeasures(pred_cwece, ref_cwece, dict_args=dict_args)
    value_test = cm.class_wise_expectation_calibration_error()
    expected_cwece = 0.150
    print(value_test)
    assert_allclose(value_test, expected_cwece, atol=0.001)


def test_gamma_ik():
    pred = [[0.1, 0.8, 0, 0.1], [0.6, 0.1, 0, 0.7], [0.3, 0.1, 1, 0.2]]
    pred = np.asarray(pred).T
    ref = np.asarray([1, 0, 2, 1])
    cm = CalibrationMeasures(pred, ref)
    value_test = cm.gamma_ik(0, 0)
    expected_gamma = gamma(1.2)
    assert_allclose(value_test, expected_gamma, atol=0.001)


def test_dirichlet_kernel():
    pred = [[0.1, 0.8, 0, 0.1], [0.6, 0.1, 0, 0.7], [0.3, 0.1, 1, 0.2]]
    pred = np.asarray(pred).T
    ref = np.asarray([1, 0, 2, 1])
    cm = CalibrationMeasures(pred, ref)
    numerator = gamma(1.2 + 2.2 + 1.6)
    denominator = gamma(1.2) * gamma(2.2) * gamma(1.6)
    prod = np.power(0.8, 0.2) * np.power(0.1, 1.2) * np.power(0.1, 0.6)
    value_test = cm.dirichlet_kernel(1, 0)
    expected_dir = numerator * prod / denominator
    assert_allclose(value_test, expected_dir, atol=0.001)


def test_ece_kde():
    pred = [[0.1, 0.8, 0, 0.1], [0.6, 0.1, 0, 0.7], [0.3, 0.1, 1, 0.2]]
    pred = np.asarray(pred).T
    ref = np.asarray([1, 0, 2, 1])
    cm = CalibrationMeasures(pred, ref)
    dir_01 = cm.dirichlet_kernel(0, 1)
    dir_02 = cm.dirichlet_kernel(0, 2)
    dir_03 = cm.dirichlet_kernel(0, 3)
    dir_10 = cm.dirichlet_kernel(1, 0)
    dir_12 = cm.dirichlet_kernel(1, 2)
    dir_13 = cm.dirichlet_kernel(1, 3)
    dir_20 = cm.dirichlet_kernel(2, 0)
    dir_21 = cm.dirichlet_kernel(2, 1)
    dir_23 = cm.dirichlet_kernel(2, 3)
    dir_30 = cm.dirichlet_kernel(3, 0)
    dir_31 = cm.dirichlet_kernel(3, 1)
    dir_32 = cm.dirichlet_kernel(3, 2)

    den_0 = dir_01 + dir_02 + dir_03
    vect_0 = dir_01 * pred[1, :] + dir_02 * pred[2, :] + dir_03 * pred[3, :]
    vect_0norm = vect_0 / den_0 - pred[0, :]

    den_1 = dir_10 + dir_12 + dir_13
    vect_1 = dir_10 * pred[0, :] + dir_12 * pred[2, :] + dir_13 * pred[3, :]
    vect_1norm = vect_1 / den_1 - pred[1, :]

    den_2 = dir_20 + dir_21 + dir_23
    vect_2 = dir_20 * pred[0, :] + dir_21 * pred[1, :] + dir_23 * pred[3, :]
    vect_2norm = vect_2 / den_2 - pred[2, :]

    den_3 = dir_30 + dir_31 + dir_32
    vect_3 = dir_30 * pred[0, :] + dir_31 * pred[1, :] + dir_32 * pred[2, :]
    vect_3norm = vect_3 / den_3 - pred[3, :]

    norm_v0 = np.sqrt(np.sum(np.square(vect_0norm)))
    norm_v1 = np.sqrt(np.sum(np.square(vect_1norm)))
    norm_v2 = np.sqrt(np.sum(np.square(vect_2norm)))
    norm_v3 = np.sqrt(np.sum(np.square(vect_3norm)))

    expected_ece_kde = np.mean([norm_v0, norm_v1, norm_v2, norm_v3])
    value_test = cm.kernel_based_ece()
    assert_allclose(value_test, expected_ece_kde, atol=0.001)
