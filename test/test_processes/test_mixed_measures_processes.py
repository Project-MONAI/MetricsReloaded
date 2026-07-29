#import pytest
from MetricsReloaded.processes.mixed_measures_processes import (
    MultiLabelLocSegPairwiseMeasure as MLIS, MultiLabelPairwiseMeasures as MLPM,
    MultiLabelLocMeasures as MLLM
)
import numpy as np
from numpy.testing import assert_allclose

# Data for panoptic quality Figure 3.51 p96
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

def test_od_percase():
    ref = [[pq_ref1, pq_ref2, pq_ref3],[pq_ref1, pq_ref3,]]
    pred = [[pq_pred1, pq_pred2, pq_pred3, pq_pred4],[pq_pred1, pq_pred3]]
    mllm = MLLM(
        [[0, 1, 2, 1],[0,0]],
        ref_class=[[0, 1, 2],[0,0]],
        pred_loc=pred,
        ref_loc=ref,
        pred_prob=[np.asarray([[0.9,0.1,0], [0.2,0.7,0.1],[0.2,0.2,0.6],[0.2,0.7,0.1]]),np.asarray([[0.7,0.2,0.1], [0.8,0,0.2]])],
        list_values=[0, 1,2],
        per_case=True,
        localization="mask_iou",
        measures_mt=["auroc"],
        measures_pcc=["fbeta"],
    )
    value_tmp1, value_tmp2 = mllm.per_label_dict()
    print(value_tmp2, value_tmp1)
    value_test = np.asarray(value_tmp1[value_tmp1["label"] == 1]["fbeta"])[0]
    value_test2 = np.asarray(value_tmp2[value_tmp2["label"] == 1]["auroc"])[0]
    expected_fbeta1 = 0
    expected_auroc1 = 0
    assert_allclose(value_test, expected_fbeta1, atol=0.001)
    assert_allclose(value_test2, expected_auroc1, atol=0.001)

def test_od_nocase():
    ref = [[pq_ref1, pq_ref2, pq_ref3]]
    pred = [[pq_pred1, pq_pred2, pq_pred3, pq_pred4]]
    mllm = MLLM(
        [[0, 1, 2, 1]],
        ref_class=[[0, 1, 2]],
        pred_loc=pred,
        ref_loc=ref,
        pred_prob=[np.asarray([[0.9,0.1,0], [0.2,0.7,0.1],[0.2,0.2,0.6],[0.2,0.7,0.1]])],
        list_values=[0, 1,2],
        per_case=False,
        localization="mask_iou",
        measures_mt=["auroc"],
        measures_pcc=["fbeta"],
    )
    value_tmp1, value_tmp2 = mllm.per_label_dict()
    print(value_tmp2, value_tmp1)
    value_test = np.asarray(value_tmp1[value_tmp1["label"] == 1]["fbeta"])[0]
    value_test2 = np.asarray(value_tmp2[value_tmp2["label"] == 1]["auroc"])[0]
    expected_fbeta1 = 0
    expected_auroc1 = 0
    assert_allclose(value_test, expected_fbeta1, atol=0.001)
    assert_allclose(value_test2, expected_auroc1, atol=0.001)

def test_mlpm_nocase():
    ref = [[pq_ref1, pq_ref2, pq_ref3],[pq_ref1, pq_ref2,]]
    pred = [[pq_pred1, pq_pred2, pq_pred3, pq_pred4],[pq_pred1, pq_pred2]]
    mlis = MLIS(
        [[1, 1, 1, 1],[0,0]],
        ref_class=[[1, 1, 1],[0,0]],
        pred_loc=pred,
        ref_loc=ref,
        pred_prob=[np.asarray([[0,1], [0,1],[0,1],[0,1]]),np.asarray([[1,0], [1,0]])],
        list_values=[0, 1],
        per_case=False,
        localization="mask_iou",
        measures_detseg=["PQ"],
        measures_pcc=["fbeta"],
    )
    value_tmp1, value_tmp2, value_tmp3 = mlis.per_label_dict()
    print(value_tmp2)
    value_test = np.asarray(value_tmp2[value_tmp2["label"] == 1]["PQ"])[0]
    expected_pq = 0.350
    assert_allclose(value_test, expected_pq, atol=0.001)

def test_create_nifti():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS(
        [[1, 1, 1, 1]],
        ref_class=[[1, 1, 1]],
        pred_loc=[pred],
        ref_loc=[ref],
        pred_prob=[np.asarray([[0,1], [0,1], [0,1], [0,1]])],
        list_values=[1],
        localization="mask_iou",
        measures_detseg=["PQ"],
        file=['examples/PredictionPQ.nii.gz'],
        flag_map=True
    )
    _, value_tmp, _ = mlis.per_label_dict()
    print(value_tmp, ' is mlis per label in PQ')
    value_test = np.asarray(value_tmp[value_tmp["label"] == 1]["PQ"])[0]
    print("PQ ", value_test)
    expected_pq = 0.350
    assert_allclose(value_test, expected_pq, atol=0.001)


def test_create_map_png():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS(
        [[1, 1, 1, 1]],
        ref_class=[[1, 1, 1]],
        pred_loc=[pred],
        ref_loc=[ref],
        pred_prob=[np.asarray([[0,1], [0,1], [0,1], [0,1]])],
        list_values=[1],
        localization="mask_iou",
        measures_detseg=["PQ"],
        file=['examples/PredictionPQ.png'],
        flag_map=True
    )
    _, value_tmp, _ = mlis.per_label_dict()
    print(value_tmp, ' is mlis per label in PQ')
    value_test = np.asarray(value_tmp[value_tmp["label"] == 1]["PQ"])[0]
    print("PQ ", value_test)
    expected_pq = 0.350
    assert_allclose(value_test, expected_pq, atol=0.001)




def test_mismatch_category():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS(
        [[0, 0, 0, 0]],
        ref_class=[[0, 1, 1]],
        pred_loc=[pred],
        ref_loc=[ref],
        pred_prob=[np.asarray([[1,0], [1,0],[1,0],[1,0]])],
        list_values=[0, 1],
        localization="mask_iou",
        measures_detseg=["PQ"],
        measures_pcc=["fbeta"],
    )
    value_tmp1, value_tmp2, value_tmp3 = mlis.per_label_dict()
    value_test = np.asarray(value_tmp2[value_tmp2["label"] == 1]["PQ"])[0]

    assert value_test == 0

def test_empty_ref_pred_pq():
    ref = []
    pred = []
    mlis = MLIS(pred_class=[[]],
                ref_class=[[]],
                pred_loc=[pred],
                ref_loc = [ref],
                pred_prob=[None],
                list_values=[1],
                measures_detseg=['PQ'],
                localization='mask_iou')
   
    _, value_tmp, _ = mlis.per_label_dict()
    value_test = np.asarray(value_tmp[value_tmp["label"] == 1]["PQ"])[0]
    print(value_tmp)
    assert value_test != value_test

# def test_name_list_pq():
#     ref = [pq_ref1, pq_ref2, pq_ref3]
#     pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
#     mlis = MLIS(
#         [[1, 1, 1, 1]],
#         ref_class=[[1, 1, 1]],
#         pred_loc=[pred],
#         ref_loc=[ref],
#         pred_prob=[np.asarray([[0,1], [0,1], [0,1], [0,1]])],
#         list_values=[1],
#         localization="mask_iou",
#         measures_detseg=["PQ"],
#     )
#     _, value_tmp, _ = mlis.per_label_dict()
#     print(value_tmp, ' is mlis per label in PQ')
#     value_test = np.asarray(value_tmp[value_tmp["label"] == 1]["PQ"])[0]
#     print("PQ ", value_test)
#     expected_pq = 0.350
#     assert mlis.names[0] == 'CasePQ'


def test_panoptic_quality():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS(
        [[1, 1, 1, 1]],
        ref_class=[[1, 1, 1]],
        pred_loc=[pred],
        ref_loc=[ref],
        pred_prob=[np.asarray([[0,1], [0,1], [0,1], [0,1]])],
        list_values=[1],
        localization="mask_iou",
        measures_detseg=["PQ"],
    )
    _, value_tmp, _ = mlis.per_label_dict()
    print(value_tmp, ' is mlis per label in PQ')
    value_test = np.asarray(value_tmp[value_tmp["label"] == 1]["PQ"])[0]
    print("PQ ", value_test)
    expected_pq = 0.350
    assert_allclose(value_test, expected_pq, atol=0.001)

def test_image_level_classification():
    pred = [[1,1]]
    ref = [[1,0]]
    pred_proba= [[[0.2,0.8],[0.4,0.6]]]
    mlpm = MLPM(pred, ref, pred_proba,[1],measures_pcc=['fbeta'], measures_calibration=['ls'],per_case=True)
    df_pcc, df_mt = mlpm.per_label_dict()
    df_mcc, df_cal = mlpm.multi_label_res()
    print(float(np.asarray(df_cal['ls'])[0]))
    value_test = float(np.asarray(df_cal['ls'])[0])
    assert_allclose(value_test, -0.57, atol=0.01)

def test_image_level_classification_withmcc():
    pred = [[1,1],[0,1]]
    ref = [[1,0],[0,0]]
    pred_proba= [[[0.2,0.8],[0.4,0.6]],[[0.8,0.2],[0.4,0.6]]]
    #pred_proba= [[[0.2,0.8],[0.4,0.6]]]
    mlpm = MLPM(pred, ref, pred_proba,list_values=[0,1],measures_pcc=['fbeta'], measures_calibration=['ls'],measures_mcc=['mcc'])
    df_pcc, df_mt = mlpm.per_label_dict()
    df_mcc, df_cal = mlpm.multi_label_res()
    print(float(np.asarray(df_cal['ls'])[0]))
    value_test = float(np.asarray(df_cal['ls'])[0])
    print(df_mcc)
    value_test2 = df_mcc['mcc']
    assert_allclose(value_test2, 0.33,atol=0.01)
    assert_allclose(value_test, -0.57, atol=0.01)

def test_image_level_classification_percase():
    pred = [[1,1],[0,1]]
    ref = [[1,0],[0,0]]
    pred_proba= [[[0.2,0.8],[0.4,0.6]],[[0.8,0.2],[0.4,0.6]]]
    mlpm = MLPM(pred, ref, pred_proba,[0,1],measures_pcc=['fbeta'], measures_calibration=['ls'],measures_mt=['auroc'])
    df_pcc, df_mt = mlpm.per_label_dict()
    df_mcc, df_cal = mlpm.multi_label_res()
    print(float(np.asarray(df_cal['ls'])[0]))
    value_test = float(np.asarray(df_cal['ls'])[0])
    assert_allclose(value_test, -0.57, atol=0.01)

def test_semanticsegmentation_twolabels():
    ref = pq_ref1 + pq_ref2+2*pq_ref3
    pred = pq_pred1 + 2*pq_pred2 + 2*pq_pred3+pq_pred4
    mlpm = MLPM([pred],[ref],pred_proba=[None],list_values=[1,2],measures_overlap=['dsc'],measures_boundary=['nsd'])
    df_bin, df_mt = mlpm.per_label_dict()
    print(df_bin)
    value_test = float(np.asarray(df_bin[df_bin['label']==1]['nsd']))
    assert_allclose(value_test, 0.52,atol=0.01)






