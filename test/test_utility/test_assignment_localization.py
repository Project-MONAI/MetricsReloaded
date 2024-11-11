import pytest
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from MetricsReloaded.processes.mixed_measures_processes import (
    MultiLabelLocSegPairwiseMeasure as MLIS,
)
from MetricsReloaded.utility.assignment_localization import AssignmentMapping
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

#Data for figure 6a testing of assignment and average precision
ref6c1 = np.asarray([3,2,7,5])
ref6c2 = np.asarray([7,9,8,11])
ref6c3 = np.asarray([1,16,3,18])
ref6c4 = np.asarray([14,14,16,18])

pred6c1 = np.asarray([2,3,6,6])
pred6c2 = np.asarray([2,15,4,17])
pred6c3 = np.asarray([13,13,15,17])
pred6c4 = np.asarray([16,7,19,10])
pred6c5 = np.asarray([12,2,15,4])

pred_proba_6c = [[0.05, 0.95],[0.30,0.70],[0.20,0.80],[0.20,0.80],[0.10,0.90]]

pred_boxes_6c = [pred6c1, pred6c2, pred6c3, pred6c4, pred6c5]
ref_boxes_6c = [ref6c1, ref6c2, ref6c3, ref6c4]



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

def test_assignment_6c():
    asm1 = AssignmentMapping(pred_loc=pred_boxes_6c, ref_loc=ref_boxes_6c, pred_prob=pred_proba_6c, thresh=0.1,localization='box_iou')
    df_matching, df_fn, df_fp, list_valid = asm1.initial_mapping()
    print(asm1.matrix, df_matching, df_fp, df_fn, list_valid)
    numb_fn = df_fn.shape[0]
    numb_fp = df_fp.shape[0]
    expected_fn = 1
    expected_fp = 2
    assert expected_fn ==  numb_fn
    assert expected_fp == numb_fp
    assert_array_almost_equal(np.asarray(list_valid),np.asarray([0,1,2]))

def test_check_localization():
    ref_box = [[2,2,4,4]]
    ref_com = [[3,3]]
    pred_box = [[2,2,4,4]]
    pred_com = [[3,3]]
    ref_mask = np.zeros([14,14])
    pred_mask = np.zeros([14,14])
    ref_mask[2:5,2:5]=1
    pred_mask[2:5,2:5]=1
    ref_boxes = np.vstack([ref_box])
    ref_masks = np.asarray([ref_mask])
    ref_coms = np.vstack([ref_com])
    pred_coms = np.vstack([pred_com])
    pred_boxes = np.vstack([pred_box])
    pred_masks = np.asarray([pred_mask])
    
    am1 = AssignmentMapping(pred_masks, ref_masks, [1],'box_iou')
    am2 = AssignmentMapping(pred_masks, ref_masks, [1], 'com_dist')
    am3 = AssignmentMapping(pred_boxes, ref_boxes, [1], 'com_dist')
    am4 = AssignmentMapping(pred_coms, ref_boxes, [1], 'point_in_box')
    expected_matrix = np.asarray([[0]])
    expected_matrix2 = np.asarray([[1]])
    assert_allclose(am1.matrix, expected_matrix2)
    assert_allclose(am2.matrix, expected_matrix)
    assert_allclose(am3.matrix, expected_matrix)
    assert_allclose(am4.matrix, expected_matrix2)

def test_check_localization_notusable():
    ref_box = [[2,2,4,4]]
    ref_com = [[3,3]]
    pred_box = [[2,2,4,4]]
    pred_com = [[3,3]]
    ref_mask = np.zeros([14,14])
    pred_mask = np.zeros([14,14])
    ref_mask[2:5,2:5]=1
    pred_mask[2:5,2:5]=1
    ref_boxes = np.vstack([ref_box])
    ref_masks = np.asarray([ref_mask])
    ref_coms = np.vstack([ref_com])
    pred_coms = np.vstack([pred_com])
    pred_boxes = np.vstack([pred_box])
    pred_masks = np.asarray([pred_mask])
    
    am1 = AssignmentMapping(pred_coms, ref_masks, [1],'box_iou')
    am2 = AssignmentMapping(pred_coms, ref_masks, [1], 'mask_com')
    am3 = AssignmentMapping(pred_boxes, ref_boxes, [1], 'mask_iou')
    am4 = AssignmentMapping(pred_coms, ref_boxes, [1], 'point_in_mask')
    am5 = AssignmentMapping(pred_masks, ref_boxes, [1], 'point_in_mask')
    am6 = AssignmentMapping(pred_masks, ref_coms, [1], 'point_in_box')
    am7 = AssignmentMapping(pred_coms, ref_coms, [1], 'point_in_box')
    
    expected_flag = False
    
    assert_allclose(am1.flag_usable, expected_flag)
    assert_allclose(am2.flag_usable, expected_flag)
    assert_allclose(am3.flag_usable, expected_flag)
    assert_allclose(am4.flag_usable, expected_flag)   
    assert_allclose(am5.flag_usable, expected_flag)
    assert_allclose(am6.flag_usable, expected_flag)
    assert_allclose(am7.flag_usable, expected_flag)

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

def test_pairwise_boxior():
    box_ref1 = np.asarray([2,2,4,4])
    box_ref2 = np.asarray([4,5,7,9])
    box_pred1 = np.asarray([2,2, 4,4])
    box_pred2 = np.asarray([9,9,10,10])
    ref_boxes = np.vstack([box_ref1, box_ref2])
    pred_boxes = np.vstack([box_pred1,box_pred2])
    print(ref_boxes)
    print(pred_boxes)
    am = AssignmentMapping(pred_boxes, ref_boxes,[1,1],'box_ior')
    
    expected_matrix = np.asarray([[1, 0],[0,0]])
    assert_allclose(am.matrix, expected_matrix)

def test_pairwise_boxcom():
    box_ref1 = np.asarray([2,2,4,4])
    box_ref2 = np.asarray([4,5,7,9])
    box_pred1 = np.asarray([2,2, 4,4])
    box_pred2 = np.asarray([9,9,10,10])
    ref_boxes = np.vstack([box_ref1, box_ref2])
    pred_boxes = np.vstack([box_pred1,box_pred2])
    print(ref_boxes)
    print(pred_boxes)
    3,3 / 5.5,7 / 3,3 /9.5/9.5
    am = AssignmentMapping(pred_boxes, ref_boxes,[1,1],'box_com')
    
    expected_matrix = np.asarray([[0, 4.72],[9.19,4.72]])
    assert_allclose(am.matrix, expected_matrix, atol=0.01)

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

def test_pairwise_pointinmask():
    ref1 = np.zeros([14,14])
    ref2 = np.zeros([14,14])
    pred1 = np.zeros([14,14])
    pred2 = np.zeros([14,14])
    ref1[2:5,2:5] = 1
    ref2[4:8,5:10] = 1
    pred1 = [3,4]
    pred2 = [9,10]
    ref_masks = np.asarray([ref1, ref2])
    pred_points = np.vstack([pred1, pred2])
    am = AssignmentMapping(pred_points, ref_masks, [1,1],'point_in_mask')
    expected_matrix = np.asarray([[1,0],[0,0]])
    assert_allclose(am.matrix, expected_matrix)

def test_pairwise_pointinbox():
    ref1 = np.asarray([2,2,4,4])
    ref2 = np.asarray([4,5,7,9])
    pred1 = [3,4]
    pred2 = [9,10]
    ref_box = np.asarray([ref1, ref2])
    pred_points = np.vstack([pred1, pred2])
    am = AssignmentMapping(pred_points, ref_box, [1,1],'point_in_box',assignment='hungarian')
    expected_matrix = np.asarray([[1,0],[0,0]])
    assert_allclose(am.matrix, expected_matrix)

def test_pairwise_pointcomdist():
    ref1 = [3,4]
    ref2 = [10,10]
    pred1 = [3,4]
    pred2 = [9,10]
    ref_com = np.vstack([ref1, ref2])
    pred_com = np.vstack([pred1, pred2])
    am = AssignmentMapping(pred_com, ref_com, [1,1],localization='com_dist')
    expected_matrix = np.asarray([[0, 9.22],[8.49, 1]])
    assert_allclose(am.matrix, expected_matrix,atol=0.01)


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
