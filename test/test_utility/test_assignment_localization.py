import pytest
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from MetricsReloaded.processes.mixed_measures_processes import (
    MultiLabelLocSegPairwiseMeasure as MLIS,
)
from MetricsReloaded.utility.assignment_localization import AssignmentMapping
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc


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


def test_pairwise_boxiou():
    box_ref1 = np.asarray([2,2,4,4])
    box_ref2 = np.asarray([4,5,7,9])
    box_pred1 = np.asarray([2,2, 4,4])
    box_pred2 = np.asarray([9,9,10,10])
    ref_boxes = np.vstack([box_ref1, box_ref2])
    pred_boxes = np.vstack([box_pred1,box_pred2])
    print(ref_boxes)
    print(pred_boxes)
    am = AssignmentMapping(pred_boxes, ref_boxes,[1,1],'box_iou')
    
    expected_matrix = np.asarray([[1, 0],[0,0]])
    assert_allclose(am.matrix, expected_matrix)

def test_pairwise_boxiou_frommask():
    ref1 = np.zeros([14,14])
    ref2 = np.zeros([14,14])
    pred1 = np.zeros([14,14])
    pred2 = np.zeros([14,14])
    ref1[2:5,2:5] = 1
    ref2[4:8,5:10] = 1
    pred1[2:5,2:5] = 1
    pred2[9:11,9:11] = 1
    ref_masks = np.asarray([ref1, ref2])
    print(ref_masks.shape)
    pred_masks = np.asarray([pred1, pred2])
    am = AssignmentMapping(pred_masks, ref_masks, [1,1],'box_iou')
    expected_matrix = np.asarray([[1,0],[0,0]])
    assert_allclose(am.matrix, expected_matrix)


def test_localization():
    ref = [f59_ref1, f59_ref2]
    pred = [f59_pred1, f59_pred2]
    mlis1 = MLIS(
        [[0, 1]],
        [[0, 1]],
        [pred],
        [ref],
        [np.asarray([[1,0],[0,1]])],
        [0, 1],
        assignment="greedy_matching",
        localization="mask_com",
        thresh=3,
    )
    mlis2 = MLIS(
        [[0, 1]],
        [[0, 1]],
        [pred],
        [ref],
        [np.asarray([[1,0], [0,1]])],
        [0, 1],
        assignment="greedy_matching",
        localization="mask_ior",
        thresh=0,
    )
    _, _, _ = mlis1.per_label_dict()
    match1 = mlis1.matching
    _, _, _ = mlis2.per_label_dict()
    match2 = mlis2.matching
    print(match1, match2, match2.columns)
    m12 = match1[match1["label"] == 1]
    m21 = match2[match2["label"] == 0]
    m22 = match2[match2["label"] == 1]
    print(m12)
    print(match1[match1["label"] == 1])
    print(match1[match1["label"] == 0])
    assert (
        np.asarray(m12[m12["pred"] == 0]["ref"])[0] == 0
        and np.asarray(m21[m21["pred"] == 0]["ref"])[0] == -1
    )
