from re import A
from pairwise_measures import BinaryPairwiseMeasures as PM
from pairwise_measures import MultiClassPairwiseMeasures as MPM
from mixed_measures_processes import MultiLabelLocSegPairwiseMeasure as MLIS
import numpy as np
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

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

f27_pred = np.concatenate([np.ones([81]),np.zeros([9]),np.ones([2]),np.zeros([8])])
f27_ref = np.concatenate([np.ones([90]),np.zeros([10])])


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


def test_ba():
    list_values = [0, 1, 2, 3]
    mpm = MPM(pred, ref, list_values)
    ohp = mpm.one_hot_pred().T
    ohr = mpm.one_hot_ref()
    cm = np.matmul(ohp, ohr)
    col_sum = np.sum(cm, 0)
    print(col_sum, np.diag(cm))
    print(np.diag(cm) / col_sum)
    numerator = np.sum(np.diag(cm) / col_sum)
    print(mpm.confusion_matrix())
    ba = mpm.balanced_accuracy()
    print(ba)
    assert ba >= 0.7071 and ba < 0.7072


def test_mcc2():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.matthews_correlation_coefficient()
    print("MCC f38 ", value_test, mcc(f38_ref, f38_pred))
    assert value_test > 0.0386 and value_test < 0.0387


def test_mcc3():
    mpm = MPM(f38_pred, f38_ref, [0, 1])
    value_test = mpm.matthews_correlation_coefficient()
    print("MCC MPM f38", value_test, mcc(f38_ref, f38_pred))
    assert value_test > 0.0386 and value_test < 0.0387


def test_cohenskappa2():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.cohens_kappa()
    print("CK f38 ", value_test, cks(f38_pred, f38_ref))
    assert np.round(value_test, 3) == 0.003

def test_expectedcost():
    bpm = PM(f27_pred, f27_ref)
    value_test = bpm.normalised_expected_cost()
    print('EC 27', value_test)
    assert np.round(value_test,2) == 0.30

def test_expectedcost2():
    mpm = MPM(f27_pred, f27_ref, [0,1])
    value_test = mpm.normalised_expected_cost()
    print('ECn', value_test)
    assert np.round(value_test,2) == 0.30

def test_cohenskappa3():
    mpm = MPM(f38_pred, f38_ref, [0, 1])
    value_test = mpm.weighted_cohens_kappa()
    print("CK f38 ", value_test, cks(f38_pred, f38_ref))
    assert np.round(value_test, 3) == 0.003


def test_balanced_accuracy2():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.balanced_accuracy()
    print("BA f38 ", value_test)
    assert np.round(value_test, 2) == 0.87


def test_youden_index2():
    bpm = PM(f38_pred, f38_ref)
    value_test = bpm.youden_index()
    print("J f38 ", value_test)
    assert np.round(value_test, 2) == 0.75


def test_mcc():
    list_values = [0, 1, 2, 3]
    mpm = MPM(pred, ref, list_values)
    mcc = mpm.matthews_correlation_coefficient()
    print(mcc)
    assert mcc < 1


def test_dsc():
    bpm = PM(p_pred, p_ref)
    print(np.sum(p_ref), np.sum(p_pred))
    value_test = bpm.fbeta()
    print("DSC test", value_test)
    assert value_test >= 0.862 and value_test < 0.8621


def test_assd():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.measured_average_distance()
    print("ASSD test", value_test)
    assert np.round(value_test, 2) == 0.44


def test_masd():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.measured_masd()
    print("MASD test", value_test)
    assert value_test == 0.425


def test_nsd():
    bpm = PM(p_pred, p_ref, dict_args={"nsd": 1})
    value_test = bpm.normalised_surface_distance()
    print("NSD 1 test ", value_test)
    assert np.round(value_test, 2) == 0.89


def test_nsd2():
    bpm = PM(p_pred, p_ref, dict_args={"nsd": 2})
    value_test = bpm.normalised_surface_distance()
    print("NSD 2 test", value_test)
    assert np.round(value_test, 1) == 1.0


def test_iou():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.intersection_over_union()
    print("IoU ", value_test)
    assert np.round(value_test, 2) == 0.76


def test_fbeta():
    pm = PM(p_large_pred1, p_large_ref)
    value_test = pm.fbeta()
    print(value_test)
    assert value_test >= 0.986 and value_test < 0.987


def test_sens():
    pm = PM(f27_pred1, f27_ref1)
    value_test = pm.sensitivity()
    print("Sensitivity ", value_test)
    assert np.round(value_test, 2) == 0.57


def test_ppv():
    print(f27_pred1, f27_ref1)
    pm = PM(f27_pred1, f27_ref1)
    value_test = pm.positive_predictive_values()
    print("PPV ", value_test)
    assert value_test > 0.975 and value_test < 0.976


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
    assert np.round(value_test1, 2) == 0.86 and np.round(value_test2, 2) == 0.67


def test_pq():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS(
        [[1, 1, 1, 1]],
        ref_class=[[1, 1, 1]],
        pred_loc=[pred],
        ref_loc=[ref],
        pred_prob=[[1, 1, 1, 1]],
        list_values=[1],
        localization="maskiou",
        measures_detseg=["PQ"],
    )
    _, value_tmp, _ = mlis.per_label_dict()
    value_test = np.asarray(value_tmp[value_tmp["label"] == 1]["PQ"])[0]
    print("PQ ", value_test)
    assert value_test > 0.350 and value_test < 0.351


def test_mismatch_category():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS(
        [[1, 1, 1, 1]],
        ref_class=[[1, 2, 2]],
        pred_loc=[pred],
        ref_loc=[ref],
        pred_prob=[[1, 1, 1, 1]],
        list_values=[1, 2],
        localization="maskiou",
        measures_detseg=["PQ"],
        measures_pcc=["fbeta"],
    )
    value_tmp1, value_tmp2, value_tmp3 = mlis.per_label_dict()
    value_test = np.asarray(value_tmp2[value_tmp2["label"] == 1]["PQ"])[0]

    assert value_test == 0


def test_empty():
    ref = [0]
    pred = [0]
    pm = PM(np.asarray(ref), np.asarray(pred))
    value_test = pm.fbeta()
    print("Empty FB ", value_test)
    assert value_test == 1


def test_localization():
    ref = [f59_ref1, f59_ref2]
    pred = [f59_pred1, f59_pred2]
    mlis1 = MLIS(
        [[1, 2]],
        [[1, 2]],
        [pred],
        [ref],
        [[1, 1]],
        [1, 2],
        assignment="Greedy matching",
        localization="maskcom",
        thresh=3,
    )
    mlis2 = MLIS(
        [[1, 2]],
        [[1, 2]],
        [pred],
        [ref],
        [[1, 1]],
        [1, 2],
        assignment="Greedy matching",
        localization="maskior",
        thresh=0,
    )
    _, _, _ = mlis1.per_label_dict()
    match1 = mlis1.matching
    _, _, _ = mlis2.per_label_dict()
    match2 = mlis2.matching
    print(match1, match2, match2.columns)
    m12 = match1[match1["label"] == 2]
    m21 = match2[match2["label"] == 1]
    m22 = match2[match2["label"] == 2]
    print(m12)
    print(match1[match1["label"] == 2])
    print(match1[match1["label"] == 1])
    assert (
        np.asarray(m12[m12["pred"] == 0]["ref"])[0] == 0
        and np.asarray(m21[m21["pred"] == 0]["ref"])[0] == -1
    )


def test_always_passes():
    assert True
