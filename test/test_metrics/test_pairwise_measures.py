import pytest
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from MetricsReloaded.processes.mixed_measures_processes import (
    MultiLabelLocSegPairwiseMeasure as MLIS,
)
import numpy as np

from MetricsReloaded.utility.utils import one_hot_encode
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures

#Data for figure SN 2.9 of Pitfalls p49
pred29_1 = np.concatenate([np.ones([45]),np.zeros([5]),np.ones([10]),np.zeros([40])])
ref29_1 = np.concatenate([np.ones([50]),np.zeros([50])])
pred29_2 = np.concatenate([np.ones([81]),np.zeros([9]),np.ones([2]),np.zeros([8])])
ref29_2 = np.concatenate([np.ones([90]),np.zeros([10])])
ppm29_1 = PM(pred29_1, ref29_1)
ppm29_2 = PM(pred29_2, ref29_2)

#Data for figure SN 2.17 of pitfalls p59
pred217_1 = np.concatenate([np.ones([6]), np.zeros([94])])
pred217_2 = np.zeros([100])
ref217 = np.concatenate([np.ones([3]),np.zeros([97])])
ppm217_1 = PM(pred217_1, ref217)
ppm217_2 = PM(pred217_2, ref217)

#Data for figure SN 2.10 of pitfalls p50
ref210 = np.zeros([14,14])
ref210[5:9,5:9] = 1
pred210_1 = np.zeros([14,14])
pred210_1[6:8,6:8] = 1
pred210_2 = np.zeros([14,14])
pred210_2[4:10,4:10]=1
ppm210_1 = PM(pred210_1, ref210)
ppm210_2 = PM(pred210_2, ref210)

#Data for figure 2.12 p53
ref212 = np.zeros([22, 22])
ref212[2:21, 2:21] = 1
pred212 = np.zeros([22, 22])
pred212[3:21, 2:21] = 1
ppm212_1 = PM(pred212, ref212)
ppm212_2 = PM(pred212,ref212,dict_args={'boundary_dist':2})

#Data for figure 5c (Hausdorff with annotation error p14 Pitfalls)
ref5c = np.zeros([14, 14])
ref5c[1, 1] = 1
ref5c[9:12, 9:12] = 1
pred5c = np.zeros([14, 14])
pred5c [9:12, 9:12] = 1
bpm5c = PM(pred5c, ref5c, dict_args={'hd_perc':95})

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

# Figure ClDice p 53 S2.14 pitfalls paper
ref214 = np.zeros([24,24])
ref214[1:10,7:12]=1
ref214[10:12,3:19]=1
ref214[12:15,3:5]=1
ref214[12:15,15:19]=1
ref214[14:20,15:17]=1
ref214[14:15,1:5]=1
ref214[14:17,2:3]=1
ref214[14:19,4:5]=1
ref214[17:18,4:8]=1
ref214[14:15,15:24]=1
ref214[12:15,22:23]=1
ref214[14:17,21:22]=1
ref214[17:20,5:6]=1
ref214[17:22,12:13]=1
ref214[19:20,12:17]=1
ref214[18:19,15:20]=1
ref214[17,19]=1

pred214_1 = np.zeros([24,24])
pred214_1[1:10,7:12]=1
pred214_1[10:12,3:15]=1

pred214_2 = np.copy(ref214)
pred214_2[10:14,3:4] = 0
pred214_2[10:11,3:9] = 0
pred214_2[10:11,10:19] = 0
pred214_2[1:11,7:9] = 0
pred214_2[1:11,10:12]=0
pred214_2[10:14,18:19]=0
pred214_2[12:14,15:17]=0
pred214_2[14:19,15:16]=0

ppm214_1 = PM(pred214_1, ref214)
ppm214_2 = PM(pred214_2, ref214)



# panoptic quality Figure 3.51 p96
pq_pred1 = np.zeros([18, 18])
pq_pred1[ 3:7,1:3] = 1
pq_pred1[3:6,3:7]=1
pq_pred2 = np.zeros([18, 18])
pq_pred2[13:16,4:6] = 1
pq_pred3 = np.zeros([18, 18])
pq_pred3[7:12,13:17] = 1
pq_pred4 = np.zeros([18, 18])
pq_pred4[13:15,13:17] = 1
pq_pred4[15,15] = 1

pq_ref1 = np.zeros([18, 18])
pq_ref1[2:7, 1:3] = 1
pq_ref1[2:5,3:6] = 1
pq_ref2 = np.zeros([18, 18])
pq_ref2[6:12,12:17] = 1
pq_ref3 = np.zeros([18, 18])
pq_ref3[14:15:,7:10] = 1
pq_ref3[13:16,8:9] = 1

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

def test_fn_map():
    """
    Using SN2.10 as basis of illustration for FP TP FN TN calculation
    """
    fn1 = ppm210_1.fn()
    fn2 = ppm210_2.fn()
    expected_fn1 = 12
    expected_fn2 = 0
    assert fn1 == 12
    assert fn2 == 0 

def test_fp():
    """
    Using SN2.10 as illustrative example for calculation of TP TN FP FN

    """
    fp1 = ppm210_1.fp()
    fp2 = ppm210_2.fp()
    expected_fp1 = 0
    expected_fp2 = 20
    assert fp1 == expected_fp1
    assert fp2 == expected_fp2

def test_tp():
    """
    Using SN2.10 as illustrative example p50 pitfalls
    """

    tp1 = ppm210_1.tp()
    tp2 = ppm210_2.tp()
    expected_tp1 = 4
    expected_tp2 = 16
    assert tp1 == expected_tp1
    assert tp2 == expected_tp2

def test_tn():
    """
    Using SN2.10 as illustrative example p50 Pitfalls paper
    """

    tn1 = ppm210_1.tn()
    tn2 = ppm210_2.tn()
    expected_tn1 = 180
    expected_tn2 = 160
    assert tn1 == expected_tn1
    assert tn2 == expected_tn2

def test_n_pos_ref():
    expected_n_pos_ref = 16
    n_pos_ref = ppm210_1.n_pos_ref()
    assert expected_n_pos_ref == n_pos_ref

def test_n_union():
    expected_n_union1 = 16
    expected_n_union2 = 36
    n_union1 = ppm210_1.n_union()
    n_union2 = ppm210_2.n_union()
    assert expected_n_union1 == n_union1
    assert expected_n_union2 == n_union2

def test_n_intersection():
    expected_n_intersection1 = 4
    expected_n_intersection2 = 16
    n_intersection1 = ppm210_1.n_intersection()
    n_intersection2 = ppm210_2.n_intersection()
    assert expected_n_intersection1 == n_intersection1
    assert expected_n_intersection2 == n_intersection2

def test_n_neg_ref():
    expected_n_neg_ref = 180
    n_neg_ref = ppm210_1.n_neg_ref()
    assert expected_n_neg_ref == n_neg_ref

def test_n_pos_pred():
    expected_n_pos_pred1 = 4
    expected_n_pos_pred2 = 36
    n_pos_pred1 = ppm210_1.n_pos_pred()
    n_pos_pred2 = ppm210_2.n_pos_pred()
    assert expected_n_pos_pred2 == n_pos_pred2
    assert expected_n_pos_pred1 == n_pos_pred1

def test_n_neg_pred():
    expected_n_neg_pred1 = 192
    expected_n_neg_pred2 = 160
    n_neg_pred1 = ppm210_1.n_neg_pred()
    n_neg_pred2 = ppm210_2.n_neg_pred()

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
    value_test = bpm.absolute_volume_difference_ratio()
    expected_vdiff = 0
    assert_allclose(value_test, expected_vdiff)

def test_specificity():
    """
    Using figure 2.17 p59 as example test
    """
    value_test1 = ppm217_1.specificity()
    value_test2 = ppm217_2.specificity()
    expected_spec1 = 0.97
    expected_spec2 = 1.00
    assert_allclose(value_test1, expected_spec1, atol=0.01)
    assert_allclose(value_test2, expected_spec2, atol=0.01)


def test_matthews_correlation_coefficient_29():
    """
    Taking SN 3.9 as figure illustration for MCC p49 Pitfalls
    """
    expected_mcc1 = 0.70
    expected_mcc2 = 0.56
    value_test1 = ppm29_1.matthews_correlation_coefficient()
    value_test2 = ppm29_2.matthews_correlation_coefficient()
    assert_allclose(value_test1, expected_mcc1, atol=0.01)
    assert_allclose(value_test2, expected_mcc2, atol=0.02)


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

def test_confusion_matrix():
    """
    Taking Figure SN3.39 as inspiration
    """

    mask_ref = np.zeros([5,6])
    mask_ref[0:5,0:4] = 1
    mask_pred = np.zeros([5,6])
    mask_pred[1:4,1:6]=1
    mpm = MPM(np.reshape(mask_pred,[-1]),np.reshape(mask_ref,[-1]),[0,1])
    cm_test = mpm.confusion_matrix()
    cm = np.asarray([[4,11],[6,9]])
    print(cm_test)
    assert_array_equal(cm_test,cm)


def test_ec3():
    test_true = np.asarray([0, 1, 2, 3, 4])
    test_pred = np.asarray([0, 1, 2, 3, 0])
    mpm = MPM(test_pred, test_true, [0, 1, 2, 3, 4])
    value_test = mpm.normalised_expected_cost()
    print(value_test)
    expected_ec = 0.25
    assert_allclose(value_test, expected_ec, atol=0.01)

def test_accuracy():
    """
    Taking as reference figure SN 2.11 p51 of Pitfalls paper
    """
    ref1 = np.concatenate([np.ones([30]), np.zeros([75])])
    pred1 = np.ones([105])
    ref2 = np.concatenate([np.ones([35]),np.zeros([70])])
    pred2 = np.concatenate([np.ones([20]),np.zeros([15]),np.ones([60]),np.zeros([10])])
    expected_accuracy1 = 0.286
    expected_accuracy2 = 0.286
    ppm1 = PM(pred1, ref1)
    ppm2 = PM(pred2, ref2)
    value_test1 = ppm1.accuracy()
    value_test2 = ppm2.accuracy()
    assert_allclose(value_test1, expected_accuracy1,atol=0.001)
    assert_allclose(value_test2, expected_accuracy2,atol=0.001)

def test_netbenefit():
    """
    Taking as reference figure SN 2.11 p 51 of Pitfalls paper
    """
    ref1 = np.concatenate([np.ones([30]), np.zeros([75])])
    pred1 = np.ones([105])
    ref2 = np.concatenate([np.ones([35]),np.zeros([70])])
    pred2 = np.concatenate([np.ones([20]),np.zeros([15]),np.ones([60]),np.zeros([10])])
    ppm = PM(pred1, ref1, dict_args={"exchange_rate": 1.0 / 9.0})
    value_test = ppm.net_benefit_treated()
    ppm2 = PM(pred2, ref2, dict_args={"exchange_rate": 1.0 / 9.0})
    value_test2 = ppm2.net_benefit_treated()
    ppm3 = PM(pred2, ref2)
    value_test3 = ppm3.net_benefit_treated()
    ppm4 = PM(pred1,ref1)
    value_test4 = ppm4.net_benefit_treated()
    print(value_test, value_test2)
    expected_netbenefit1 = 0.206
    expected_netbenefit2 = 0.127
    expected_netbenefit3 = -0.381
    expected_netbenefit4 = -0.429
    assert_allclose(value_test, expected_netbenefit1, atol=0.001)
    assert_allclose(value_test2, expected_netbenefit2, atol=0.001)
    assert_allclose(value_test3, expected_netbenefit3, atol=0.001)
    assert_allclose(value_test4, expected_netbenefit4, atol=0.001)

def test_cohenskappa2():

    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.cohens_kappa()
    print("CK f38 ", value_test, cks(f38_pred, f38_ref))
    expected_ck = 0.003
    assert_allclose(value_test, expected_ck, atol=0.001)


def test_negative_predictive_value():
    """
    Taking figure SN 2.9 as inspiration p49 Pitfalls
    """
    value_test1 = ppm29_1.negative_predictive_value()
    value_test2 = ppm29_2.negative_predictive_value()
    expected_npv1 = 0.889
    expected_npv2 = 0.47
    assert_allclose(value_test1, expected_npv1, atol=0.001)
    assert_allclose(value_test2, expected_npv2, atol=0.01)
    


def test_expectedcost():
    bpm = PM(f27_pred, f27_ref)
    value_test = bpm.normalised_expected_cost()
    print("EC 27", value_test)
    expected_ec = 0.30
    assert_allclose(value_test, expected_ec, atol=0.01)


def test_normalised_expectedcost2():
    """
    TAking SN 3.9 as reference p49 pitfalls
    """
    value_test1 = ppm29_1.normalised_expected_cost()
    value_test2 = ppm29_2.normalised_expected_cost()
    expected_ec = 0.30
    
    assert_allclose(value_test2, expected_ec, atol=0.01)
    assert_allclose(value_test1, expected_ec, atol=0.01)
    
def test_cohenskappa():
    """
    Taking SN 2.9 p49 Pitfalls as reference
    """
    value_test1 = ppm29_1.cohens_kappa()
    value_test2 = ppm29_2.cohens_kappa()
    expected_ck1 = 0.70
    expected_ck2 = 0.53
    assert_allclose(value_test1, expected_ck1, atol=0.01)
    assert_allclose(value_test2, expected_ck2, atol=0.01)


def test_cohenskappa3():
    mpm = MPM(f38_pred, f38_ref, [0, 1])
    value_test = mpm.weighted_cohens_kappa()
    print("CK f38 ", value_test, cks(f38_pred, f38_ref))
    expected_ck3 = 0.003
    assert_allclose(value_test, expected_ck3, atol=0.001)


def test_balanced_accuracy2():
    """
    Taking Figure SN 2.39 as inspiration p49 pitfalls
    """
    expected_ba1 = 0.85
    expected_ba2 = 0.85
    value_test1 = ppm29_1.balanced_accuracy()
    value_test2 = ppm29_2.balanced_accuracy()
    assert_allclose(value_test1, expected_ba1, atol=0.01)
    assert_allclose(value_test2, expected_ba2, atol=0.01)


def test_youden_index2():
    """
    Taking as inspiration figure SN2.9 p49 Pitfalls
    """
    expected_yi1 = 0.70
    expected_yi2 = 0.70
    value_test1 = ppm29_1.youden_index()
    value_test2 = ppm29_2.youden_index()
    assert_allclose(value_test1, expected_yi1, atol=0.01)
    assert_allclose(value_test2, expected_yi2, atol=0.01)

def test_mcc():
    list_values = [0, 1, 2, 3]
    mpm = MPM(pred, ref, list_values)
    mcc = mpm.matthews_correlation_coefficient()
    print(mcc)
    assert mcc < 1



# def test_distance_empty():
#     """
#     Testing that output is 0 when reference and prediction empty for calculation of distance
#     """
#     pred = np.zeros([14, 14])
#     ref = np.zeros([14, 14])
#     bpm = PM(pred, ref)
#     value_test = bpm.measured_distance()
#     expected_dist = (0, 0, 0, 0)
#     assert_allclose(value_test, expected_dist)

def test_fbeta():
    """
    Taking inspiration from SN 2.9 - p49 Pitfalls
    """
    expected_f11 = 0.86
    expected_f12 = 0.94
    value_test1 = ppm29_1.fbeta()
    value_test2 = ppm29_2.fbeta()
    assert_allclose(value_test1, expected_f11, atol=0.01)
    assert_allclose(value_test2, expected_f12, atol=0.01)

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


def test_intersection_over_union():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.intersection_over_union()
    print("IoU ", value_test)
    expected_iou = 0.76
    assert_allclose(value_test, expected_iou, atol=0.01)


def test_fbeta_beta_value():
    """
    Taking inspiration from SN 2.9 - p49 Pitfalls
    """
    expected_f11 = 0.86
    expected_f12 = 0.94
    ppm29_1.dict_args={'beta':1}
    ppm29_2.dict_args={'beta':1}
    value_test1 = ppm29_1.fbeta()
    value_test2 = ppm29_2.fbeta()
    assert_allclose(value_test1, expected_f11, atol=0.01)
    assert_allclose(value_test2, expected_f12, atol=0.01)


def test_sensitivity():
    """
    Using figure 2.17 p59 as example for test
    """

    value_test1 = ppm217_1.sensitivity()
    value_test2 = ppm217_2.sensitivity()
    expected_sens1 = 1.00
    expected_sens2 = 0.00
    assert_allclose(value_test1, expected_sens1, atol=0.01)
    assert_allclose(value_test2, expected_sens2, atol=0.01)

def test_recall():
    """
    Using figure 2.17 p59 as example for test
    """

    value_test1 = ppm217_1.recall()
    value_test2 = ppm217_2.recall()
    expected_rec1 = 1.00
    expected_rec2 = 0.00
    assert_allclose(value_test1, expected_rec1, atol=0.01)
    assert_allclose(value_test2, expected_rec2, atol=0.01)

def test_sens():
    pm = PM(f27_pred1, f27_ref1)
    value_test = pm.sensitivity()
    print("Sensitivity ", value_test)
    expected_sens = 0.57
    assert_allclose(value_test, expected_sens, atol=0.01)


def test_positive_predictive_value():
    """
    Taking as inspiration figure SN2.9 p49 Pitfalls
    """
    
    value_test1 = ppm29_1.positive_predictive_value()
    value_test2 = ppm29_2.positive_predictive_value()
    expected_ppv1 = 0.82
    expected_ppv2 = 0.98
    assert_allclose(value_test1, expected_ppv1, atol=0.01)
    assert_allclose(value_test2, expected_ppv2, atol=0.01)


def test_positive_likelihood_ratio():
    """
    Taking as inspiration figure SN2.9 p49 Pitfalls
    """
    ppm1 = PM(pred29_1, ref29_1)
    ppm2 = PM(pred29_2, ref29_2)
    value_test1 = ppm1.positive_likelihood_ratio()
    value_test2 = ppm2.positive_likelihood_ratio()
    expected_plr1 = 4.50
    expected_plr2 = 4.50
    assert_allclose(value_test1, expected_plr1, atol=0.01)
    assert_allclose(value_test2, expected_plr2, atol=0.01)

def test_hausdorff_distances_s210():
    """
    Using Figure 2.10 as illustrative example
    """
    hausdorff_1 = ppm210_1.measured_hausdorff_distance()
    hausdorff_2 = ppm210_2.measured_hausdorff_distance()
    expected_hd1 = 1.41
    expected_hd2 = 1.41
    assert_allclose(hausdorff_1,expected_hd1,atol=0.01)
    assert_allclose(hausdorff_2,expected_hd2,atol=0.01)

def test_masd_s210():
    """
    Using Figure 2.10 as illustrative example
    """
    masd_1 = ppm210_1.measured_masd()
    masd_2 = ppm210_2.measured_masd()
    expected_masd1 = 1.07
    expected_masd2 = 1.04
    assert_allclose(masd_1,expected_masd1,atol=0.01)
    assert_allclose(masd_2,expected_masd2,atol=0.01)

def test_assd_s210():
    assd_1 = ppm210_1.measured_average_distance()
    assd_2 = ppm210_2.measured_average_distance()
    expected_assd1 = 1.10
    expected_assd2 = 1.05
    assert_allclose(assd_1, expected_assd1,atol=0.01)
    assert_allclose(assd_2, expected_assd2,atol=0.01)

def test_nsd_s210():
    """
    Using Figure 2.10 as illustrative example
    """
    nsd_1 = ppm210_1.normalised_surface_distance()
    nsd_2 = ppm210_2.normalised_surface_distance()
    expected_nsd1 = 0.75
    expected_nsd2 = 0.875
    print(nsd_1,nsd_2)
    assert_allclose(nsd_1,expected_nsd1,atol=0.01)
    assert_allclose(nsd_2,expected_nsd2,atol=0.01)

def test_hausdorff_distance_5c():
    """
    Using figure 5c p14 as illustration for calculation of HD and HD95
    """
    hausdorff_distance = bpm5c.measured_hausdorff_distance()
    hausdorff_distance_perc = bpm5c.measured_hausdorff_distance_perc()
    print(hausdorff_distance_perc)
    expected_hausdorff_distance = 11.31
    expected_hausdorff_distance_perc = 6.79
    assert_allclose(hausdorff_distance, expected_hausdorff_distance, atol=0.01)
    assert_allclose(
        hausdorff_distance_perc, expected_hausdorff_distance_perc, atol=0.01
    )

def test_distance_empty_ref():
    ppm1 = PM(pred29_1, ref29_1*0)
    hd, hd_perc, masd, assd = ppm1.measured_distance()
    assert np.isnan(hd)
    assert np.isnan(hd_perc)
    assert np.isnan(masd)
    assert np.isnan(assd)

def test_distance_empty_pred():
    ppm1 = PM(pred29_1*0, ref29_1)
    hd, hd_perc, masd, assd = ppm1.measured_distance()
    assert np.isnan(hd)
    assert np.isnan(hd_perc)
    assert np.isnan(masd)
    assert np.isnan(assd)


def test_distance_empty_pred_and_ref():
    ppm1 = PM(pred29_1*0, ref29_1*0)
    hd, hd_perc, masd, assd = ppm1.measured_distance()
    assert np.isnan(hd)
    assert np.isnan(hd_perc)
    assert np.isnan(masd)
    assert np.isnan(assd)

def test_calculate_worse_dist():
    ppm_pix1 = PM(pred210_1, ref210)
    ppm_pix12 = PM(pred210_1, ref210,pixdim=[1,2])
    assert_allclose(ppm_pix1.worse_dist,19.80,atol=0.01)
    assert_allclose(ppm_pix12.worse_dist,31.30,atol=0.01)


def test_boundary_iou():
    """
    Taking as inspiration figure S 2.12
    """
    
    value_test1 = ppm212_1.boundary_iou()
    expected_biou_1 = 0.6
    expected_biou_2 = 0.8
    value_test2 = ppm212_2.boundary_iou()
    assert_allclose(value_test1, expected_biou_1, atol=0.1)
    assert_allclose(value_test2, expected_biou_2, atol=0.1)

def test_empty_ref_pred_nsd_biou():
    ref_empty = np.zeros([14,14])
    pred_empty = np.zeros([14,14])
    ppm_empty = PM(pred_empty, ref_empty)
    nsd = ppm_empty.normalised_surface_distance()
    assert np.isnan(nsd)
    biou = ppm_empty.boundary_iou()
    assert np.isnan(biou)
    cldsc = ppm_empty.centreline_dsc()
    assert np.isnan(cldsc)

    
def test_cldsc_s214():
    value_test1 = ppm214_1.centreline_dsc()
    value_test2 = ppm214_2.centreline_dsc()
    expected_cldsc1 = 0.475
    expected_cldsc2 = 0.78
    assert_allclose(value_test1, expected_cldsc1, atol=0.01)
    assert_allclose(value_test2, expected_cldsc2, atol=0.01)

def test_dsc_s214():
    value_test1 = ppm214_1.dsc()
    value_test2 = ppm214_2.dsc()
    expected_dsc1 = 0.666
    expected_dsc2 = 0.685
    assert_allclose(value_test1, expected_dsc1, atol=0.01)
    assert_allclose(value_test2, expected_dsc2, atol=0.01)

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
    assert np.isnan(fbeta)
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
