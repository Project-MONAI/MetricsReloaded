import pytest
from metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from processes.mixed_measures_processes import MultiLabelLocSegPairwiseMeasure as MLIS
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

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
        localization="mask_iou",
        measures_detseg=["PQ"],
        measures_pcc=["fbeta"],
    )
    value_tmp1, value_tmp2, value_tmp3 = mlis.per_label_dict()
    value_test = np.asarray(value_tmp2[value_tmp2["label"] == 1]["PQ"])[0]

    assert value_test == 0



def test_panoptic_quality():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS(
        [[1, 1, 1, 1]],
        ref_class=[[1, 1, 1]],
        pred_loc=[pred],
        ref_loc=[ref],
        pred_prob=[[1, 1, 1, 1]],
        list_values=[1],
        localization="mask_iou",
        measures_detseg=["PQ"],
    )
    _, value_tmp, _ = mlis.per_label_dict()
    value_test = np.asarray(value_tmp[value_tmp["label"] == 1]["PQ"])[0]
    print("PQ ", value_test)
    expected_pq = 0.350
    assert_allclose(value_test, expected_pq, atol=0.001)
    
