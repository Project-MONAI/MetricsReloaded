from MetricsReloaded.utility.utils import intersection_boxes, box_ior, box_iou, area_box, union_boxes
import numpy as np
from numpy.testing import assert_allclose

def test_area_box():
    box1 = [2,2,2,2,3,3]
    expected_vol = 4
    value_test = area_box(np.asarray(box1))
    assert_allclose(value_test, expected_vol, atol=0.01)

def test_union_boxes():
    box1 = [2,2,2,2,3,3]
    box2 = [2,2,2,3,4,4]
    expected_vol = 18
    value_test = union_boxes(np.asarray(box1),np.asarray(box2))
    assert_allclose(value_test, expected_vol)

def test_intersection_boxes():
    box1 = [2,2,2,2,3,3]
    box2 = [2,2,2,3,4,4]
    expected_vol = 4
    value_test = intersection_boxes(np.asarray(box1), np.asarray(box2))
    assert_allclose(value_test, expected_vol)

def test_box_iou():
    box1 = [2,2,2,2,3,3]
    box2 = [2,2,2,3,4,4]
    expected_iou = 0.222
    value_test = box_iou(np.asarray(box1), np.asarray(box2))
    assert_allclose(value_test, expected_iou, atol=0.001)

def test_box_ior():
    box1 = [2,2,2,2,3,3]
    box2 = [2,2,2,3,4,4]
    expected_ior = 0.222
    value_test = box_ior(np.asarray(box1), np.asarray(box2))
    assert_allclose(value_test, expected_ior, atol=0.001)
