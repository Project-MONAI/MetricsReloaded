import pytest
from metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from processes.mixed_measures_processes import MultiLabelLocSegPairwiseMeasure as MLIS
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
        assignment="greedy_matching",
        localization="mask_com",
        thresh=3,
    )
    mlis2 = MLIS(
        [[1, 2]],
        [[1, 2]],
        [pred],
        [ref],
        [[1, 1]],
        [1, 2],
        assignment="greedy_matching",
        localization="mask_ior",
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
