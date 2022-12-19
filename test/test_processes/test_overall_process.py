from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from MetricsReloaded.processes.mixed_measures_processes import (
    MultiLabelLocSegPairwiseMeasure as MLIS, MultiLabelPairwiseMeasures as MLPM,
)
from MetricsReloaded.processes.overall_process import ProcessEvaluation as PE
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

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

data_init = {}
data_init['pred_class'] = [pred1, pred2]
data_init['ref_class'] = [ref1, ref2]
data_init['list_values'] = [1]
data_init['pred_prob'] = [None,None]

data_miss = {}
data_miss['pred_class'] = [pred1, pred2]
data_miss['ref_class'] = [ref1, ref2]
data_miss['list_values'] = [1]
data_miss['pred_prob'] = [None,None]
data_miss['ref_missing'] = [ref3]

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


def test_op_aggregation():
    pe = PE(data_init,'Semantic Segmentation',measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab)
    assert_allclose(pe.grouped_lab.shape,[2,4])


def test_op_refmissing():
    pe = PE(data_miss,'Semantic Segmentation',measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab, pe.resseg)
    assert_allclose(pe.grouped_lab.shape,[3,4])



def test_op_agg_label():
    pe = PE(data_agg, category='Semantic Segmentation', measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab)
    assert_allclose(pe.grouped_lab.shape, [1,9])

def test_op_agg_label_nan():
    pe = PE(data_agg2, category="Semantic Segmentation", measures_overlap=['fbeta'],measures_boundary=['boundary_iou'])
    print(pe.grouped_lab, pe.resseg)
    assert_allclose(pe.grouped_lab.shape, [2,8])