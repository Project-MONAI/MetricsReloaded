from MetricsReloaded.processes.overall_process import ProcessEvaluation as PE
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


ref1 = np.zeros([21,21])
ref1[5:12,4:7] = 1
ref2 = np.zeros([21,21])
ref2[14:18,12:16] = 1
ref3 = np.zeros([21,21])
ref3[1:4,13:15] = 1

pred1 = np.zeros([21,21])
pred1[8:14,6:8] =1
pred2 = np.zeros([21,21])
pred2[15:17,13:15] = 1
ref12 = ref1 + 2*ref2
pred12 = pred1 + 2*pred2

pred_empty = np.zeros([21,21])
ref_empty = np.zeros([21,21])

data_init = {}
data_init['pred_class'] = [pred1, pred2]
data_init['ref_class'] = [ref1, ref2]
data_init['list_values'] = [1]
data_init['pred_prob'] = [None,None]

data_init_ml = {}
data_init_ml['pred_class'] = [pred1, pred12]
data_init_ml['ref_class'] = [ref1, ref12]
data_init_ml['list_values'] = [1,2]
data_init_ml['pred_prob'] = [None,None]
data_init_ml['names'] = ['Case1','Case2']

data_ilc = {}
data_ilc['pred_class'] = [np.reshape(pred12,[-1])]
data_ilc['ref_class'] = [np.reshape(ref12,[-1])]
data_ilc['list_values'] = [0,1,2]
data_ilc['pred_prob'] = [None]

data_nan = {}
data_nan['pred_class'] = [pred1, pred1, pred_empty, pred1]
data_nan['ref_class'] = [ref12, ref1, ref1, ref_empty]
data_nan['list_values'] = [1,2]
data_nan['pred_prob'] = [None,None,None,None]

data_miss = {}
data_miss['pred_class'] = [pred1, pred2]
data_miss['ref_class'] = [ref1, ref2]
data_miss['list_values'] = [1]
data_miss['pred_prob'] = [None,None]
data_miss['ref_missing_pred'] = [ref3]

data_agg = {}
data_agg['pred_class'] = [pred12]
data_agg['ref_class'] = [ref12]
data_agg['list_values'] = [1,2]
data_agg['pred_prob'] = [None,None]

data_agg2 = {}
data_agg2['pred_class'] = [pred12,pred1]
data_agg2['ref_class'] = [ref12,ref1]
data_agg2['list_values'] = [1,2]
data_agg2['pred_prob'] = [None,None]
data_agg2['weight_labels'] = [1,3]

data_empty_od = {}


#Data from Panoptic Quality - 3.51 p96 of Pitfalls
#Figure 3.51 p96
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

ref_351 = [pq_ref1, pq_ref2, pq_ref3]
pred_351 = [pq_pred1, pq_pred2, pq_pred3,pq_pred4]

data_pq = {}
for k in ['pred_loc','ref_loc','pred_class','ref_class','pred_prob','weight_labels','file','ref_missing_pred']:
    data_pq[k] = []
data_pq['pred_loc'] = [pred_351]
data_pq['ref_loc'] = [ref_351]
data_pq['list_values'] = [1]
data_pq['pred_prob'] = [np.asarray([[0.5,0.5], [0.6,0.4], [0.3,0.7],[0.3,0.7]])]
data_pq['ref_class'] = [[1, 1, 1]]
data_pq['pred_class']= [[1,1,1,1]]
data_pq['weight_labels'] = [1]



#Data for figure 6a 
ref6a1 = np.asarray([3,2,7,5])
ref6a2 = np.asarray([7,9,8,11])
ref6a3 = np.asarray([1,16,3,18])
ref6a4 = np.asarray([14,14,16,18])

pred6a1 = np.asarray([2,3,6,6])
pred6a2 = np.asarray([2,15,4,17])
pred6a3 = np.asarray([13,13,15,17])
pred6a4 = np.asarray([16,7,19,10])
pred6a5 = np.asarray([12,2,15,4])

pred_proba_6a = [[0.05, 0.95],[0.70,0.30],[0.20,0.80],[0.20,0.80],[0.10,0.90]]
pred_class_6a = [0, 0, 1, 1, 1]
ref_class_6a = [0, 0, 1, 1]
pred_boxes_6a = [pred6a1, pred6a2, pred6a3, pred6a4, pred6a5]
ref_boxes_6a = [ref6a1, ref6a2, ref6a3, ref6a4]

data_od = {}
for k in ['pred_loc','ref_loc','pred_class','ref_class','pred_prob','weight_labels','file','ref_missing_pred']:
    data_od[k] = []
data_od['pred_loc'] = [pred_boxes_6a]
data_od['ref_loc'] = [ref_boxes_6a]
data_od['list_values'] = [0,1]
data_od['pred_class'] = [pred_class_6a]
data_od['ref_class'] = [ref_class_6a]
data_od['pred_prob'] = [np.asarray(pred_proba_6a)]
data_od['weight_labels'] = {0:1,1:3}


data_od_empty_ref = {}
for k in ['pred_loc','ref_loc','pred_class','ref_class','pred_prob','weight_labels','file','ref_missing_pred']:
    data_od_empty_ref[k] = []
data_od_empty_ref['pred_loc'] = [pred_boxes_6a,pred_boxes_6a]
data_od_empty_ref['ref_loc'] = [ref_boxes_6a,[]]
data_od_empty_ref['list_values'] = [0,1]
data_od_empty_ref['pred_class'] = [pred_class_6a,pred_class_6a]
data_od_empty_ref['ref_class'] = [ref_class_6a,[]]
data_od_empty_ref['pred_prob'] = [np.asarray(pred_proba_6a),np.asarray(pred_proba_6a)]
data_od_empty_ref['weight_labels'] = {0:1,1:3}

pred_com_351 = [np.asarray([4.5,2]), np.asarray([14,5]), np.asarray([9,14.5]), np.asarray([13.5,14.5])]

def test_non_valid_task():
    pe = PE(data_nan,'SemSeg',measures_overlap=['fbeta'],measures_boundary=['masd'])
    assert not pe.flag_valid

def test_ilc():
    pe = PE(data_ilc,'ImLC', measures_mcc=['ec','mcc'],measures_pcc=[])
    print(pe.stats_all)
    assert_array_equal(pe.stats_all.columns,['ec','mcc','case'])

def test_is():
    pe = PE(data_pq, 'InS', measures_pcc=['fbeta'],measures_overlap=['dsc'])
    print(pe.resseg)
    assert pe.resseg.shape[0] == 2

def test_od():
    pe = PE(data_od, 'ObD',measures_pcc=['fbeta'],localization='box_iou',thresh_ass=0.1)
    print(pe.resdet)
    assert pe.resdet.shape[0] == 2

def test_od_emptyref():
    pe = PE(data_od_empty_ref, 'ObD', measures_pcc=['sensitivity','fbeta'],localization='box_iou',thresh_ass=0.1)
    print(pe.resdet)
    assert pe.resdet.shape[0] == 4

def test_op_nanreplacement():
    pe = PE(data_nan,'SemS',measures_overlap=['fbeta'],measures_boundary=['masd'])
    print(pe.resseg, pe.resseg.columns,pe.resseg[['fbeta','fbeta_nanrep','masd','masd_nanrep','check_empty','label','case']])
    assert_allclose(pe.resseg.shape,[8,10])
    df_test = pe.resseg
    assert_allclose(df_test[(df_test['label']==1) & (df_test['case']==2)]['masd_nanrep'],29.69,atol=0.01)

def test_op_aggregation():
    pe = PE(data_init,'SemS',measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab)
    assert_allclose(pe.grouped_lab.shape,[2,8])


def test_op_refmissingpred():
    pe = PE(data_miss,'SemS',measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab, pe.resseg)
    assert_allclose(pe.grouped_lab.shape,[3,8])

def test_op_agg_label():
    pe = PE(data_agg, category='SemS', measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab)
    assert_allclose(pe.grouped_lab.shape, [1,9])

def test_op_agg_label_nan():
    pe = PE(data_agg2, category="SemS", measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab, pe.resseg)
    assert_allclose(pe.grouped_lab.shape, [2,9])

def test_check_measures_cat_valid():
    pe = PE(data_agg,category='ImLC',measures_mt=['froc'])
    assert not pe.flag_valid 