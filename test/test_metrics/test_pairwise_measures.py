import pytest
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from MetricsReloaded.processes.mixed_measures_processes import (
    MultiLabelLocSegPairwiseMeasure as MLIS,
)
import numpy as np
from MetricsReloaded.utility.utils import one_hot_encode
from numpy.testing import assert_allclose
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures

### Small size of structures relative to pixel/voxel size (DSC)
## Larger structure
p_large_ref = np.zeros((11, 11))
p_large_ref[2:5, 2:9] = 1
p_large_ref[5:9, 2:6] = 1

p_large_pred1 = np.zeros((11, 11))
p_large_pred1[2:5, 2:9] = 1
p_large_pred1[5:9, 2:6] = 1
p_large_pred1[8, 2] = 0


p_large_pred2 = np.zeros((11, 11))
p_large_pred2[2:5, 2:9] = 1
p_large_pred2[5:9, 2:6] = 1
p_large_pred2[7:9, 2] = 0

# Figure 10 for distance metrics
p_ref = np.zeros([11, 11])
p_ref[3:8, 3:8] = 1
p_pred = np.zeros([11, 11])
p_pred[3:8, 3:8] = 1
p_pred[5, 1:10] = 1
p_pred[1:10, 5] = 1

# Figure 27 a
f27_ref1 = np.concatenate([np.ones([70]), np.zeros([1])])
f27_pred1 = np.concatenate([np.ones([40]), np.zeros([30]), np.ones([1])])
f27_ref2 = f27_pred1
f27_pred2 = f27_ref1

# panoptic quality
pq_pred1 = np.zeros([21, 21])
pq_pred1[5:7, 2:5] = 1
pq_pred2 = np.zeros([21, 21])
pq_pred2[14:18, 4:6] = 1
pq_pred2[16, 3] = 1
pq_pred3 = np.zeros([21, 21])
pq_pred3[14:18, 7:12] = 1
pq_pred4 = np.zeros([21, 21])
pq_pred4[2:8, 13:16] = 1
pq_pred4[2:4, 12] = 1

pq_ref1 = np.zeros([21, 21])
pq_ref1[8:11, 3] = 1
pq_ref1[9, 2:5] = 1
pq_ref2 = np.zeros([21, 21])
pq_ref2[14:19, 7:13] = 1
pq_ref3 = np.zeros([21, 21])
pq_ref3[2:7, 14:17] = 1
pq_ref3[2:4, 12:14] = 1

f27_pred = np.concatenate([np.ones([81]), np.zeros([9]), np.ones([2]), np.zeros([8])])
f27_ref = np.concatenate([np.ones([90]), np.zeros([10])])


f38_pred = np.concatenate([np.ones([1499]), np.zeros([501])])
f38_ref = np.concatenate([np.ones([1999]), np.zeros([1])])

pred = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
]
ref = [
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    4,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    4,
    1,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    1,
    1,
    2,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
]
pred = np.asarray(pred) - 1
ref = np.asarray(ref) - 1

# Data for Figure 43 of pitfalls
f43_refl = np.zeros([21, 21])
f43_refl[1, 1:20] = 1
f43_refl[1:20, 1] = 1
f43_refl[1:17, 2] = 1
f43_refl[2, 1:17] = 1

f43_predl1 = np.copy(f43_refl)
f43_predl1[19, 1] = 0
f43_predl2 = np.copy(f43_predl1)
f43_predl2[18, 1] = 0

f43_refs = np.zeros([21, 21])
f43_refs[16:20, 1:4] = 1
f43_refs[18:20, 2:4] = 0
f43_preds1 = np.copy(f43_refs)
f43_preds1[19, 1] = 0
f43_preds2 = np.copy(f43_preds1)
f43_preds2[18, 1] = 0

ref_clDice_large = np.zeros((20, 20))
ref_clDice_large[1:4, 1:2] = 1
ref_clDice_large[4:19, 1:3] = 1
ref_clDice_large[17:19, 1:16] = 1
ref_clDice_large[18:19, 16:19] = 1

pred_clDice_large1 = np.zeros((20, 20))
pred_clDice_large1[2:4, 1:2] = 1
pred_clDice_large1[4:19, 1:3] = 1
pred_clDice_large1[17:19, 1:16] = 1
pred_clDice_large1[18:19, 16:19] = 1

pred_clDice_large2 = np.zeros((20, 20))
pred_clDice_large2[3:4, 1:2] = 1
pred_clDice_large2[4:19, 1:3] = 1
pred_clDice_large2[17:19, 1:16] = 1
pred_clDice_large2[18:19, 16:19] = 1

## Small structure
ref_clDice_small = np.zeros((14, 14))
ref_clDice_small[1:4, 1:2] = 1
ref_clDice_small[3:4, 1:4] = 1

pred_clDice_small1 = np.zeros((14, 14))
pred_clDice_small1[2:4, 1:2] = 1
pred_clDice_small1[3:4, 1:4] = 1

pred_clDice_small2 = np.zeros((14, 14))
pred_clDice_small2[3:4, 1:4] = 1

## Data for figure 59 and testing of localisation
f59_ref1 = np.zeros([15, 15])
f59_ref1[0:14, 1:13] = 1
f59_ref1[1:13, 2:12] = 0
f59_ref2 = np.zeros([15, 15])
f59_ref2[7:9, 6:10] = 1
f59_pred1 = np.zeros([15, 15])
f59_pred1[7:9, 8:10] = 1
f59_pred2 = np.zeros([15, 15])
f59_pred2[4:8, 5:9] = 1


def test_balanced_accuracy():
    list_values = [0, 1, 2, 3]
    mpm = MPM(pred, ref, list_values)
    ohp = one_hot_encode(mpm.pred, 4).T
    ohr = one_hot_encode(mpm.ref, 4)
    cm = np.matmul(ohp, ohr)
    col_sum = np.sum(cm, 0)
    numerator = np.sum(np.diag(cm) / col_sum)
    ba = mpm.balanced_accuracy()
    expected_ba = 0.7071
    assert_allclose(ba, expected_ba, atol=0.001)


def test_voldiff():
    ref = np.zeros([14, 14])
    pred = np.zeros([14, 14])
    ref[2:4, 2:4] = 1
    pred[3:5, 3:5] = 1
    bpm = PM(pred, ref)
    value_test = bpm.vol_diff()
    expected_vdiff = 0
    assert_allclose(value_test, expected_vdiff)


def test_matthews_correlation_coefficient():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.matthews_correlation_coefficient()
    print("MCC f38 ", value_test, mcc(f38_ref, f38_pred))
    expected_mcc = 0.0386
    mpm = MPM(f38_pred, f38_ref, [0, 1])
    value_test2 = mpm.matthews_correlation_coefficient()
    print("MCC MPM f38", value_test, mcc(f38_ref, f38_pred))

    assert_allclose(value_test, expected_mcc, atol=0.001)
    assert_allclose(value_test2, expected_mcc, atol=0.001)


def test_ec3():
    test_true = np.asarray([0, 1, 2, 3, 4])
    test_pred = np.asarray([0, 1, 2, 3, 0])
    mpm = MPM(test_pred, test_true, [0, 1, 2, 3, 4])
    value_test = mpm.normalised_expected_cost()
    print(value_test)
    expected_ec = 0.25
    assert_allclose(value_test, expected_ec, atol=0.01)


def test_netbenefit():
    ref = np.concatenate([np.ones([30]), np.zeros([75])])
    pred = np.ones([105])
    pred2 = np.concatenate(
        [np.ones([22]), np.zeros([8]), np.ones([50]), np.zeros([25])]
    )
    ppm = PM(pred, ref, dict_args={"exchange_rate": 1.0 / 9.0})
    value_test = ppm.net_benefit_treated()
    ppm2 = PM(pred2, ref, dict_args={"exchange_rate": 1.0 / 9.0})
    value_test2 = ppm2.net_benefit_treated()
    ppm3 = PM(pred, ref)
    value_test3 = ppm3.net_benefit_treated()
    print(value_test, value_test2)
    expected_netbenefit1 = 0.206
    expected_netbenefit2 = 0.157
    expected_netbenefit3 = -0.429
    assert_allclose(value_test, expected_netbenefit1, atol=0.001)
    assert_allclose(value_test2, expected_netbenefit2, atol=0.001)
    assert_allclose(value_test3, expected_netbenefit3, atol=0.001)


def test_cohenskappa2():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.cohens_kappa()
    print("CK f38 ", value_test, cks(f38_pred, f38_ref))
    expected_ck = 0.003
    assert_allclose(value_test, expected_ck, atol=0.001)


def test_negative_predictive_value():
    f17_ref = np.concatenate([np.ones([50]), np.zeros([50])])
    f17_pred = np.concatenate(
        [np.ones([45]), np.zeros([5]), np.ones([10]), np.zeros(40)]
    )
    bpm = PM(f17_pred, f17_ref)
    value_test = bpm.negative_predictive_values()
    expected_npv = 0.889
    assert_allclose(value_test, expected_npv, atol=0.001)
    print("NPV", value_test)


def test_expectedcost():
    bpm = PM(f27_pred, f27_ref)
    value_test = bpm.normalised_expected_cost()
    print("EC 27", value_test)
    expected_ec = 0.30
    assert_allclose(value_test, expected_ec, atol=0.01)


def test_expectedcost2():
    mpm = MPM(f27_pred, f27_ref, [0, 1])
    value_test = mpm.normalised_expected_cost()
    print("ECn", value_test)
    expected_ec = 0.30
    assert_allclose(value_test, expected_ec, atol=0.01)


def test_cohenskappa3():
    mpm = MPM(f38_pred, f38_ref, [0, 1])
    value_test = mpm.weighted_cohens_kappa()
    print("CK f38 ", value_test, cks(f38_pred, f38_ref))
    expected_ck3 = 0.003
    assert_allclose(value_test, expected_ck3, atol=0.001)


def test_balanced_accuracy2():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.balanced_accuracy()
    print("BA f38 ", value_test)
    expected_ba = 0.87
    assert_allclose(value_test, expected_ba, atol=0.01)


def test_youden_index2():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.youden_index()
    print("J f38 ", value_test)
    expected_yi = 0.75
    assert_allclose(value_test, expected_yi, atol=0.01)


def test_mcc():
    list_values = [0, 1, 2, 3]
    mpm = MPM(pred, ref, list_values)
    mcc = mpm.matthews_correlation_coefficient()
    print(mcc)
    assert mcc < 1


def test_distance_empty():
    pred = np.zeros([14, 14])
    ref = np.zeros([14, 14])
    bpm = PM(pred, ref)
    value_test = bpm.measured_distance()
    expected_dist = (0, 0, 0, 0)
    assert_allclose(value_test, expected_dist)


def test_dsc_fbeta():
    bpm = PM(p_pred, p_ref)
    bpm2 = PM(p_pred, p_ref, dict_args={"fbeta": 2})
    print(np.sum(p_ref), np.sum(p_pred))
    value_test = bpm.fbeta()
    print("DSC test", value_test)
    expected_dsc = 0.862
    value_test2 = bpm.dsc()
    
    assert_allclose(value_test, expected_dsc, atol=0.001)
    assert_allclose(value_test2, expected_dsc, atol=0.001)


def test_assd():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.measured_average_distance()
    print("ASSD test", value_test)
    expected_assd = 0.44
    assert_allclose(value_test, expected_assd, atol=0.01)


def test_masd():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.measured_masd()
    print("MASD test", value_test)
    expected_masd = 0.425
    assert_allclose(value_test, expected_masd, atol=0.001)


def test_nsd():
    bpm = PM(p_pred, p_ref, dict_args={"nsd": 1})
    value_test = bpm.normalised_surface_distance()
    print("NSD 1 test ", value_test)
    expected_nsd = 0.89
    assert_allclose(value_test, expected_nsd, atol=0.01)


def test_nsd2():
    bpm = PM(p_pred, p_ref, dict_args={"nsd": 2})
    value_test = bpm.normalised_surface_distance()
    print("NSD 2 test", value_test)
    expected_nsd2 = 1.0
    assert_allclose(value_test, expected_nsd2, atol=0.01)


def test_iou():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.intersection_over_union()
    print("IoU ", value_test)
    expected_iou = 0.76
    assert_allclose(value_test, expected_iou, atol=0.01)


def test_fbeta():
    pm = PM(p_large_pred1, p_large_ref)
    pm2 = PM(p_large_pred1, p_large_ref, dict_args={"beta": 1})
    value_test = pm.fbeta()
    value_test2 = pm2.fbeta()
    print(value_test)
    expected_fbeta = 0.986
    assert_allclose(value_test, expected_fbeta, atol=0.001)
    assert_allclose(value_test2, expected_fbeta, atol=0.001)


def test_sens():
    pm = PM(f27_pred1, f27_ref1)
    value_test = pm.sensitivity()
    print("Sensitivity ", value_test)
    expected_sens = 0.57
    assert_allclose(value_test, expected_sens, atol=0.01)


def test_ppv():
    print(f27_pred1, f27_ref1)
    pm = PM(f27_pred1, f27_ref1)
    value_test = pm.positive_predictive_values()
    print("PPV ", value_test)
    expected_ppv = 0.975
    assert_allclose(value_test, expected_ppv, atol=0.001)


def test_positive_likelihood_ratio():
    f17_ref = np.concatenate([np.ones([50]), np.zeros([50])])
    f17_pred = np.concatenate(
        [np.ones([45]), np.zeros([5]), np.ones([10]), np.zeros(40)]
    )
    bpm = PM(f17_pred, f17_ref)
    value_test = bpm.positive_likelihood_ratio()
    expected_plr = 4.5
    assert_allclose(value_test, expected_plr, atol=0.01)


def test_hd():
    f20_ref = np.zeros([14, 14])
    f20_ref[1, 1] = 1
    f20_ref[9:12, 9:12] = 1
    f20_pred = np.zeros([14, 14])
    f20_pred[9:12, 9:12] = 1
    bpm = PM(f20_pred, f20_ref, dict_args={"hd_perc": 95})
    hausdorff_distance = bpm.measured_hausdorff_distance()
    hausdorff_distance_perc = bpm.measured_hausdorff_distance_perc()

    expected_hausdorff_distance = 11.31
    expected_hausdorff_distance_perc = 6.22
    assert_allclose(hausdorff_distance, expected_hausdorff_distance, atol=0.01)
    assert_allclose(
        hausdorff_distance_perc, expected_hausdorff_distance_perc, atol=0.01
    )


def test_boundary_iou():
    f21_ref = np.zeros([22, 22])
    f21_ref[2:21, 2:21] = 1
    f21_pred = np.zeros([22, 22])
    f21_pred[3:21, 2:21] = 1
    bpm = PM(f21_pred, f21_ref)
    value_test = bpm.boundary_iou()
    expected_biou = 0.6
    assert_allclose(value_test, expected_biou, atol=0.1)
    assert np.round(value_test, 1) == 0.6


def test_cldsc():
    pm1 = PM(pred_clDice_small1, ref_clDice_small)
    value_test1 = pm1.centreline_dsc()
    pm2 = PM(pred_clDice_small2, ref_clDice_small)
    value_test2 = pm2.centreline_dsc()
    print(
        "clDSC small1",
        value_test1,
        np.sum(pred_clDice_small1),
        np.sum(ref_clDice_small),
    )
    print(
        "clDSC small2",
        value_test2,
        np.sum(pred_clDice_small2),
        np.sum(ref_clDice_small),
    )
    expected_cldsc1 = 0.86
    expected_cldsc2 = 0.67
    assert_allclose(value_test1, expected_cldsc1, atol=0.01)
    assert_allclose(value_test2, expected_cldsc2, atol=0.01)


def test_empty_reference():
    ref = [0]
    pred = [0]
    pm = PM(np.asarray(pred), np.asarray(ref))

    match = "reference is empty, recall not defined"
    with pytest.warns(UserWarning, match=match):
        fbeta = pm.fbeta()

    match2 = "reference empty, sensitivity not defined"
    with pytest.warns(UserWarning, match=match2):
        sens = pm.sensitivity()

    match3 = "reference all positive, specificity not defined"
    ref2 = [1]
    pred2 = [1]
    pm2 = PM(np.asarray(pred2), np.asarray(ref2))
    with pytest.warns(UserWarning, match=match3):
        spec = pm2.specificity()

    expected_fbeta = 1
    assert fbeta == expected_fbeta
    assert sens != sens  # True if nan
    assert spec != spec  # True if nan


def test_pred_in_ref():
    pred = np.zeros([14, 14])
    ref = np.zeros([14, 14])
    pred[2:4, 2:4] = 1
    ref[3:6, 3:6] = 1
    bpm = PM(pred, ref)
    expected_pir = 1
    value_test = bpm.pred_in_ref()
    assert value_test == expected_pir
    ref[3, 3] = 0
    bpm2 = PM(pred, ref)
    expected_pir2 = 0
    value_test2 = bpm2.pred_in_ref()
    assert value_test2 == expected_pir2


def test_com_empty():
    pred0 = np.zeros([14, 14])
    ref0 = np.zeros([14, 14])
    bpm = PM(pred0, ref0)
    value_test = bpm.com_dist()
    value_pred = bpm.com_pred()
    value_ref = bpm.com_ref()
    expected_empty = -1
    assert value_test == expected_empty
    assert value_ref == expected_empty
    assert value_pred == expected_empty


def test_com_dist():
    pred = np.zeros([14, 14])
    ref = np.zeros([14, 14])
    pred[0:5, 0:5] = 1
    ref[0:5, 0:5] = 1
    bpm = PM(pred, ref)
    value_dist = bpm.com_dist()
    value_pred = bpm.com_pred()
    value_ref = bpm.com_ref()
    expected_dist = 0
    expected_com = (2, 2)
    assert_allclose(value_pred, expected_com)
    assert_allclose(value_ref, expected_com)
    assert_allclose(value_dist, expected_dist, atol=0.01)
