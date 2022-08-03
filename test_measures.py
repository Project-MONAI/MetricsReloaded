from pairwise_measures import BinaryPairwiseMeasures as PM
from pairwise_measures import MultiClassPairwiseMeasures as MPM
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


def test_iou():
    pm = PM(p_large_pred1, p_large_ref)
    value_test = pm.fbeta()
    print(value_test)
    assert value_test >= 0.986 and value_test < 0.987

def test_always_passes():
    assert True
