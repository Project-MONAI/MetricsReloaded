from pairwise_measures import BinaryPairwiseMeasures as PM
from pairwise_measures import MultiClassPairwiseMeasures as MPM
from mixed_measures_processes import MultiLabelLocSegPairwiseMeasure as MLIS
import numpy as np

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

#Figure 10 for distance metrics
p_ref = np.zeros([11,11])
p_ref[3:8,3:8] = 1
p_pred =np.zeros([11,11])
p_pred[3:8,3:8]=1
p_pred[5,1:10] =1
p_pred[1:10,5]=1

# Figure 27 a
f27_ref1 = np.concatenate([np.ones([70]),np.zeros([1])])
f27_pred1 = np.concatenate([np.ones([40]), np.zeros([30]),np.ones([1])])
f27_ref2 = f27_pred1
f27_pred2 = f27_ref1

#panoptic quality
pq_pred1 = np.zeros([21,21])
pq_pred1[5:7,2:5] = 1
pq_pred2 = np.zeros([21,21])
pq_pred2[14:18,4:6] = 1
pq_pred2[16,3] =1
pq_pred3 = np.zeros([21,21])
pq_pred3[14:18,7:12] = 1
pq_pred4 = np.zeros([21,21])
pq_pred4[2:8,13:16] =1
pq_pred4[2:4,12]=1

pq_ref1 = np.zeros([21,21])
pq_ref1[8:11,3] =1
pq_ref1[9,2:5] = 1
pq_ref2 = np.zeros([21,21])
pq_ref2[14:19,7:13] =1
pq_ref3 = np.zeros([21,21])
pq_ref3[2:7,14:17] = 1
pq_ref3[2:4,12:14] = 1




pred = [1, 1, 1, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
ref = [1,1,1,1,1,1,2,2,2,3,4,2,2,2,2,2,2,2,2,2,4,4, 1,2,3,3,3,3,3,3,3,3,3,3,4,1,1,2,3,3,4,4,4,4,4,4,4,4,4,4,4,4]
pred = np.asarray(pred)-1
ref = np.asarray(ref)-1


def test_ba():
    list_values = [0,1,2,3]
    mpm = MPM(pred, ref, list_values)
    ohp = mpm.one_hot_pred().T
    ohr = mpm.one_hot_ref()
    cm = np.matmul(ohp,ohr)
    col_sum = np.sum(cm, 0)
    print(col_sum, np.diag(cm))
    print(np.diag(cm)/col_sum)
    numerator = np.sum(np.diag(cm)/col_sum)
    print(mpm.confusion_matrix())
    ba = mpm.balanced_accuracy()
    print(ba)
    assert ba >= 0.7071 and ba <0.7072

def test_mcc():
    list_values = [0,1,2,3]
    mpm = MPM(pred, ref, list_values)
    mcc = mpm.matthews_correlation_coefficient()
    print(mcc)
    assert mcc<1

def test_dsc():
    bpm = PM(p_pred, p_ref)
    print(np.sum(p_ref), np.sum(p_pred))
    value_test = bpm.fbeta()
    print('DSC test', value_test)
    assert value_test >=0.862 and value_test<0.8621

def test_assd():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.measured_average_distance()
    print('ASSD test', value_test)
    assert np.round(value_test,2) == 0.44

def test_masd():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.measured_masd()
    print('MASD test', value_test)
    assert value_test == 0.425

def test_nsd():
    bpm = PM(p_pred, p_ref,dict_args={'nsd':1})
    value_test = bpm.normalised_surface_distance()
    print('NSD 1 test ', value_test)
    assert np.round(value_test,2) == 0.89

def test_nsd2():
    bpm = PM(p_pred, p_ref, dict_args={'nsd':2})
    value_test = bpm.normalised_surface_distance()
    print('NSD 2 test', value_test)
    assert np.round(value_test,1) == 1.0

def test_iou():
    bpm = PM(p_pred, p_ref)
    value_test = bpm.intersection_over_union()
    print('IoU ',value_test)
    assert np.round(value_test,2) == 0.76

def test_fbeta():
    pm = PM(p_large_pred1, p_large_ref)
    value_test = pm.fbeta()
    print(value_test)
    assert value_test >= 0.986 and value_test < 0.987

def test_sens():
    pm = PM(f27_pred1, f27_ref1)
    value_test = pm.sensitivity()
    print('Sensitivity ', value_test)
    assert np.round(value_test,2) == 0.57

def test_ppv():
    print(f27_pred1, f27_ref1)
    pm = PM(f27_pred1, f27_ref1)
    value_test = pm.positive_predictive_values()
    print('PPV ', value_test)
    assert value_test > 0.975 and value_test<0.976

def test_pq():
    ref = [pq_ref1, pq_ref2, pq_ref3]
    pred = [pq_pred1, pq_pred2, pq_pred3, pq_pred4]
    mlis = MLIS([[1,1,1,1]], ref_class=[[1,1,1]], pred_loc=[pred],ref_loc=[ref], pred_prob=[[1,1,1,1]],list_values = [1], localization='maskiou', measures_detseg=['PQ'])
    _, value_tmp, _ = mlis.per_label_dict()
    value_test = np.asarray(value_tmp[value_tmp['label']==1]['PQ'])[0]
    print('PQ ', value_test)
    assert value_test == 0.35





def test_always_passes():
    assert True
