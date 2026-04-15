
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import precision_recall_curve as prc
from MetricsReloaded.utility.utils import trapezoidal_integration
from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures as PPM


def test_fn_thresh():
    ref = np.asarray([0, 0, 0, 1, 1, 1])
    pred_proba = np.asarray([0.21, 0.35, 0.63, 0.92, 0.32, 0.79])
    ppm = PPM(pred_proba, ref)
    value_test = ppm.fn_thr(0.75)
    expected_fn75 = 1
    assert value_test == expected_fn75

def test_all_multi_threshold_values_limited_thresholds():
    ref = np.zeros([100])   
    ref[50:] = 1
    pred = np.zeros([100])
    pred[0:30] = 0.2
    pred[30:50] = 0.45
    pred[50:80] = 0.65
    pred[80:] = 0.90

    ppm = PPM(pred, ref)
    list_sens_exp = [0, 0.4, 1, 1, 1] 
    list_spec_exp = [1, 1, 1, 0.6, 0]
    list_ppv_exp = [1, 1, 1, 0.71, 0.5]
    unique_thresh = [1.9, 0.9, 0.65, 0.45, 0.2]

    t, se, sp, pp, fp = ppm.all_multi_threshold_values()

    assert_allclose(np.asarray(t), np.asarray(unique_thresh))
    assert_allclose(np.asarray(se), np.asarray(list_sens_exp), atol=0.01)
    assert_allclose(np.asarray(sp), np.asarray(list_spec_exp), atol=0.01)
    assert_allclose(np.asarray(pp), np.asarray(list_ppv_exp), atol=0.01)

def test_all_multi_threshold_values_limited_samples():
    ref = np.zeros([200])   
    ref[100:] = 1
    pred = np.zeros([200])
    pred[0:60] = 0.2
    pred[60:100] = 0.45
    pred[100:160] = 0.65
    pred[160:] = 0.90

    ppm = PPM(pred, ref)
    list_sens_exp = [0, 0.4, 1, 1, 1] 
    list_spec_exp = [1, 1, 1, 0.6, 0]
    list_ppv_exp = [1, 1, 1, 0.71, 0.5]
    unique_thresh = [1.9, 0.9, 0.65, 0.45, 0.2]

    t, se, sp, pp, fp = ppm.all_multi_threshold_values()

    assert_allclose(np.asarray(t), np.asarray(unique_thresh))
    assert_allclose(np.asarray(se), np.asarray(list_sens_exp), atol=0.01)
    assert_allclose(np.asarray(sp), np.asarray(list_spec_exp), atol=0.01)
    assert_allclose(np.asarray(pp), np.asarray(list_ppv_exp), atol=0.01)

def test_sensitivity_thr_empty_ref():
    ref = np.zeros([10])
    pred = np.arange(0,1,0.1)
    ppm = PPM(pred, ref)
    value_test = ppm.sensitivity_thr(0.4)
    assert value_test != value_test

def test_net_benefit_treated():
    ref = np.zeros([10])
    ref[5:] = 1
    pred = np.zeros([10])
    pred[4:] = 0.6
    ppm = PPM(pred, ref)
    ppm2 = PPM(pred, ref, dict_args={'benefit_proba':0.5})
    expected_value = 0.05
    value_test = ppm.net_benefit_treated()
    value_test2 = ppm2.net_benefit_treated()
    # 5/10 * 1/10 *1 - 0.5 * 0.1 * 1 = 0.05
    assert value_test == expected_value
    assert value_test2 == expected_value


def test_ppv_thr_allempty():
    ref = np.zeros([10])
    pred = np.zeros([10])
    ppm = PPM(pred, ref)
    value_test = ppm.positive_predictive_values_thr(0.2)
    assert value_test != value_test

def test_ppv_thr_ref0_thresh_moremaxpred():
    ref = np.zeros([10])
    pred = np.zeros([10])
    pred[5:] = 0.6
    ppm = PPM(pred, ref)
    value_test = ppm.positive_predictive_values_thr(0.7)
    value_test2 = ppm.positive_predictive_values_thr(0.5)
    assert value_test != value_test
    assert value_test2 == 0




def test_all_multi_threshold_large():
    ref = np.zeros([100])
    pred = np.arange(0,1,0.01)
    ppm = PPM(pred,ref)

    t, se, sp, pp, fp = ppm.all_multi_threshold_values(max_number_samples=50, max_number_thresh=10)
    expected_thresh = [1.99, 0.97, 0.86, 0.75, 0.64, 0.53, 0.42, 0.31, 0.2, 0.09, 0]
    assert_allclose(np.asarray(t), np.asarray(expected_thresh))



def test_auroc():
    """
    Based on SN2.18 p60 of Pitfalls paper
    """
    ref = np.asarray([0, 0, 0, 1, 1, 1])
    pred_proba = np.asarray([0.21, 0.35, 0.63, 0.92, 0.32, 0.79])
    ppm = PPM(pred_proba, ref)
    value_test = ppm.auroc()
    print(value_test)
    expected_auc = 0.78
    assert_allclose(value_test, expected_auc, atol=0.01)


def test_average_precision():
    """
    Based on SN2.18 p60 of pitfalls paper
    """
    ref = np.asarray([0, 0, 0, 1, 1, 1])
    pred_proba = np.asarray([0.21, 0.35, 0.63, 0.92, 0.32, 0.79])
    ppm = PPM(pred_proba, ref)
    # threshs = [0, 0.21, 0.32, 0.35, 0.63, 0.79, 0.92]
    recall = [1, 1, 0.66667, 0.66667, 0.66667, 0.33, 0]
    prec = [0.5, 0.6, 0.5, 0.66667, 1, 1, 1]
    expected_ap = trapezoidal_integration(
        np.asarray(recall)[::-1], np.asarray(prec)[::-1]
    )
    print("From SK", prc(ref, pred_proba))

    # expected_aps = aps(ref, pred_proba)
    value_test = ppm.average_precision()
    assert_allclose(value_test, expected_ap, atol=0.01)

def test_to_dict_meas():
    ref = np.asarray([0, 0, 0, 1, 1, 1])
    pred_proba = np.asarray([0.21, 0.35, 0.63, 0.92, 0.32, 0.79])
    ppm = PPM(pred_proba, ref, measures=['auroc','ap'])
    dict_ppm = ppm.to_dict_meas()
    print(dict_ppm.keys())
    assert list(dict_ppm.keys()) == ['auroc', 'ap']
    assert_allclose(dict_ppm['auroc'], 0.78, atol=0.01)

def test_sensitivity_at_specificity():
    ref = np.concatenate([np.zeros([50]), np.ones([50])])
    pred = np.arange(0, 1, 0.01)
    ppm = PPM(pred, ref)
    ppm2 = PPM(pred, ref, dict_args={'value_sensitivity':0.8, 'value_specificity':0.8,'value_ppv':0.8})
    value_sensspec = ppm.sensitivity_at_specificity()
    value_specsens = ppm.specificity_at_sensitivity()
    value_sensppv = ppm.sensitivity_at_ppv()
    value_ppvsens = ppm.ppv_at_sensitivity()

    value_sensspec2 = ppm2.sensitivity_at_specificity()
    value_specsens2 = ppm2.specificity_at_sensitivity()
    value_sensppv2 = ppm2.sensitivity_at_ppv()
    value_ppvsens2 = ppm2.ppv_at_sensitivity()

    expected_sensatspec = 1.0
    expected_specatsens = 1.0
    expected_sensatppv = 1.0
    expected_ppvatsens = 1.0
    assert_allclose(value_sensspec, expected_sensatspec, atol=0.01)
    assert_allclose(value_sensppv, expected_sensatppv, atol=0.01)
    assert_allclose(value_specsens, expected_specatsens, atol=0.01)
    assert_allclose(value_ppvsens, expected_ppvatsens, atol=0.01)

    assert_allclose(value_sensspec2, expected_sensatspec, atol=0.01)
    assert_allclose(value_sensppv2, expected_sensatppv, atol=0.01)
    assert_allclose(value_specsens2, expected_specatsens, atol=0.01)
    assert_allclose(value_ppvsens2, expected_ppvatsens, atol=0.01)

def test_fppi_thr():
    ref1 = [0, 0, 0, 1, 1, 1]
    ref2 = [0, 1, 0, 1, 0, 1]
    pred1 = [0, 0.2, 0.4, 0.6, 0.8, 1]
    pred2 = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ref = [np.asarray(ref1)], [np.asarray(ref2)]
    pred = [np.asarray(pred1)], [np.asarray(pred2)]
    ppm = PPM(pred, ref, case=np.asarray([0,1]))
    value_test = ppm.fppi_thr(0.4)
    expected_value = 1.5
    assert value_test == expected_value

def test_fppi_thr_nocase():
    ref1 = [0, 0, 0, 1, 1, 1]
    ref2 = [0, 1, 0, 1, 0, 1]
    pred1 = [0, 0.2, 0.4, 0.6, 0.8, 1]
    pred2 = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ref = np.concatenate([np.expand_dims(np.asarray(ref1),1),np.expand_dims(np.asarray(ref2),1)],1)
    pred = np.concatenate([np.expand_dims(np.asarray(pred1),1),np.expand_dims(np.asarray(pred2),1)],1)
    ppm = PPM(pred, ref)
    value_test = ppm.fppi_thr(0.4)
    expected_value = 1.5
    assert value_test == expected_value

def test_froc():
    ref = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # list_thresh [2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    list_sens = [0, 0.5, 1, 1, 1, 1, 1,1, 1, 1,1 ]
    list_fppi = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ppm = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1))
    value_test = ppm.froc()
    expected_value = trapezoidal_integration(np.asarray(list_fppi), np.asarray(list_sens))
    assert value_test == expected_value

def test_froc_above8fppi():
    ref = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1]
    # list_thresh [2, 1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    list_sens = [0, 0.5, 1, 1, 1, 1, 1,1, 1, 1,1,1]
    list_fppi = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ppm = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1))
    value_test = ppm.froc()
    expected_value = trapezoidal_integration(np.asarray(list_fppi[:-1]), np.asarray(list_sens[:-1]))
    assert value_test == expected_value

def test_froc_below8fppi():
    ref = [ 0, 0, 0, 0, 0, 0, 0, 1, 1]
    pred = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # list_thresh [2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,]
    list_sens = [0, 0.5, 1, 1, 1, 1, 1,1, 1, 1,1]
    list_fppi = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7,8]
    ppm = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1))
    value_test = ppm.froc()
    expected_value = trapezoidal_integration(np.asarray(list_fppi), np.asarray(list_sens))
    assert value_test == expected_value


def test_froc_below1over8fppi():
    ref = [ 0, 0, 0, 0, 0, 0, 0, 1, 1]
    pred = [ 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2]
    # list_thresh = [1.2, 0.2, 0.1]
    list_fppi = [0, 0, 0, 1.0/8, 1.0/4, 1.0/2, 1, 2, 4, 8]
    list_sens = [0, 0.5, 1 , 1, 1, 1,1,1,1,1]
    ppm = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1))
    value_test = ppm.froc()
    expected_value = trapezoidal_integration(np.asarray(list_fppi), np.asarray(list_sens))
    assert value_test == expected_value

def test_fppi_at_sensitivity():
    ref = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # list_thresh [2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # list_sens = [0, 0.2, 0.4, 0.6, 0.8, 1, 1,1, 1, 1,1 ]
    # list_fppi = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    ppm = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1))
    ppm2 = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1), dict_args={'value_sensitivity': 0.8})
    value_test = ppm.fppi_at_sensitivity()
    value_test2 = ppm2.fppi_at_sensitivity()
    expected_value = 5
    assert value_test == expected_value
    assert value_test2 == expected_value

def test_sensitivity_at_fppi():
    ref = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # list_thresh [2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # list_sens = [0, 0.2, 0.4, 0.6, 0.8, 1, 1,1, 1, 1,1 ]
    # list_fppi = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    ppm = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1))
    ppm2 = PPM(np.expand_dims(np.asarray(pred),1), np.expand_dims(np.asarray(ref),1), dict_args={'value_fppi': 2})
    value_test = ppm.sensitivity_at_fppi()
    value_test2 = ppm2.sensitivity_at_fppi()
    expected_value = 1
    assert value_test == expected_value
    assert value_test2 == expected_value








