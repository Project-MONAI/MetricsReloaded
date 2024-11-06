import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from MetricsReloaded.utility.utils import intersection_boxes, guess_input_style, com_from_box, point_in_box, point_in_mask, area_box, compute_box, compute_center_of_mass, compute_skeleton, combine_df, distance_transform_edt, one_hot_encode, median_heuristic, box_ior, box_iou, union_boxes, max_x_at_y_less, min_x_at_y_less, skeletonize, trapezoidal_integration


box3 = [2,3, 4,4]
box4 = [4,4,5,6]


def test_intersection_boxes_empty():
    box1 = [2,3,5,7]
    box2 = [6,8,10,10]
    intersection = intersection_boxes(box1,box2)
    assert_allclose(intersection, 0)


def test_intersection_boxes_shared_corner():
    box1 = [2,3,5,7]
    box3 = [2,3, 4,4]
    intersection = intersection_boxes(box1, box3)
    assert_allclose(intersection, 6)

def test_intersection_boxes_contained():
    box1 = [2,3,5,7]
    box4 = [4,4,5,6]
    intersection = intersection_boxes(box1, box4)
    assert_allclose(intersection, 6)

def test_guess_input_style():
    mask = np.zeros([4,5])
    mask[2:3,1:4]=1
    box = np.asarray([2,1,3,4])
    com = np.asarray([2.5,2.5])
    test_mask = guess_input_style(mask)
    test_box = guess_input_style(box)
    test_com = guess_input_style(com)
    assert test_mask == 'mask'
    assert test_box == 'box'
    assert test_com == 'com'

def test_com_from_box():
    box_1 = [2,2,3,3]
    box_2 = [1,2,1,2]
    com_1 = com_from_box(np.asarray(box_1))
    com_2 = com_from_box(np.asarray(box_2))
    assert_array_equal(com_1, np.asarray([2.5,2.5])) 
    assert_array_equal(com_2, np.asarray([1,2]))

def test_point_in_box():
    box = [2,1,5,8]
    point1 = [3,6]
    point2 = [1,9]
    assert point_in_box(np.asarray(point1), np.asarray(box)) == True
    assert point_in_box(np.asarray(point2), np.asarray(box)) == False

def test_point_in_mask():
    mask = np.zeros([10,10])
    mask[2:6,1:9] = 1
    point1 = [3,6]
    point2 = [1,9]
    assert point_in_mask(np.asarray(point1), np.asarray(mask)) == True
    assert point_in_mask(np.asarray(point2), np.asarray(mask)) == False

def test_area_box():
    box = [1,2,3,1,3,5]
    assert area_box(np.asarray(box)) == 6


# def test_compute_box():
# def test_compute_center_of_mass():
# def test_box_ior():
# def test_box_iou():
# def test_union_boxes():
# def 
# point_in_box, point_in_mask, area_box, compute_box, compute_center_of_mass, compute_skeleton, combine_df, distance_transform_edt, one_hot_encode, median_heuristic, box_ior, box_iou, union_boxes, max_x_at_y_less, min_x_at_y_less, skeletonize, trapezoidal_integration