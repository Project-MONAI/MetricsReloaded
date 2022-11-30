"""
Utility functions - :mod:`MetricsReloaded.utility.utils`
========================================================

This module provides functions for calculating common useful outputs and :ref:`morphological
<morphological>` operations.

.. _morphological:

Calculating morphological operations
------------------------------------

.. autoclass:: MorphologyOps
    :members:

Defining utility functions
--------------------------

.. currentmodule:: MetricsReloaded.utility.utils

.. autofunction:: intersection_boxes
.. autofunction:: union_boxes
.. autofunction:: area_box
.. autofunction:: box_iou
.. autofunction:: compute_center_of_mass
.. autofunction:: distance_transform_edt
.. autofunction:: max_x_at_y_more
.. autofunction:: max_x_at_y_less
.. autofunction:: min_x_at_y_more
.. autofunction:: min_x_at_y_less
.. autofunction:: trapezoidal_integration


"""
from functools import partial

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy.spatial.distance import squareform, pdist


__all__ = [
    'CacheFunctionOutput',
    'MorphologyOps',
    'intersection_boxes',
    'area_box',
    'union_boxes',
    'box_iou',
    'compute_skeleton',
    'compute_center_of_mass',
    'distance_transform_edt',
    'max_x_at_y_more',
    'max_x_at_y_less',
    'min_x_at_y_less',
    'min_x_at_y_more',
    'one_hot_encode',
    'to_string_count',
    'to_string_dist',
    'to_string_mt',
    'to_dict_meas_',
    'trapezoidal_integration',
]

class CacheFunctionOutput(object):
    """
    this provides a decorator to cache function outputs
    to avoid repeating some heavy function computations
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        return partial(self, obj)  # to remember func as self.func

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*args, **kw)
        return value


class MorphologyOps(object):
    """
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    """

    def __init__(self, binary_img, neigh):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.neigh = neigh

    def border_map(self):
        """
        Create the border map defined as the difference between the original image 
        and its eroded version

        :return: border
        """
        eroded = ndimage.binary_erosion(self.binary_map)
        border = self.binary_map - eroded
        return border

    def border_map2(self):
        """
        Creates the border for a 3D image
        :return:
        """
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def foreground_component(self):
        return ndimage.label(self.binary_map)

    def list_foreground_component(self):
        labels, _ = self.foreground_component()
        list_ind_lab = []
        list_volumes = []
        list_com = []
        list_values = np.unique(labels)
        for f in list_values:
            if f > 0:
                tmp_lab = np.where(
                    labels == f, np.ones_like(labels), np.zeros_like(labels)
                )
                list_ind_lab.append(tmp_lab)
                list_volumes.append(np.sum(tmp_lab))
                list_com.append(ndimage.center_of_mass(tmp_lab))
        return list_ind_lab, list_volumes, list_com

def intersection_boxes(box1, box2):
    """
    Intersection between two boxes given the corners

    :return: intersection 
    """
    min_values = np.minimum(box1, box2)
    max_values = np.maximum(box1, box2)
    box_inter = max_values[: min_values.shape[0] // 2]
    box_inter2 = min_values[max_values.shape[0] // 2 :]
    box_intersect = np.concatenate([box_inter, box_inter2])
    box_intersect_area = np.prod(
        np.maximum(box_inter2 + 1 - box_inter, np.zeros_like(box_inter))
    )
    return np.max([0, box_intersect_area])


def guess_input_style(a):
    """
    Given an array a, guess whether it represents a mask, a box or a centre of mass

    """
    print(a.ndim)
    if a.ndim > 1:
        return 'mask'
    else:
        if np.size(a) > 3:
            return 'box'
        else:
            return 'com'

def com_from_box(box):
    min_corner = box[:box.shape[0]//2]
    max_corner = box[box.shape[0]//2:]
    aggregate = np.vstack([min_corner,max_corner])
    com = np.mean(aggregate, 0)
    return com

def point_in_box(point, box):
    min_corner = box[:box.shape[0]//2]
    max_corner = box[box.shape[0]//2:]
    diff_min = point - min_corner
    diff_max = max_corner - point
    diff_all = np.concatenate([diff_min, diff_max])
    diff_select = diff_all[diff_all<0]
    if diff_select.size > 0 :
        return 0
    else:
        return 1

def point_in_mask(point, mask):
    new_mask = np.zeros_like(mask)
    if new_mask.ndim == 2:
        new_mask[point[0],point[1]] = 1
    else:
        new_mask[point[0],point[1],point[2]] = 1
    overlap = np.multiply(new_mask, mask)
    print(point, new_mask, overlap, 'sum_overlap ', np.sum(overlap))
    if np.sum(overlap) > 0:
        return 1
    else:
        return 0

def area_box(box1):
    """
    Determines the area / volume given the coordinates of extreme corners
    
    :param: box extreme corners specified as :math:`x_{min},y_{min},x_{max},y_{max}` or
    :math:`x_{min},y_{min},z_{min},x_{max},y_{max},z_{max}` 
    :return: area/volume of the box (in pixels/voxels)
    """
    box_corner1 = box1[: box1.shape[0] // 2]
    box_corner2 = box1[box1.shape[0] // 2 :]
    return np.prod(box_corner2 + 1 - box_corner1)


def union_boxes(box1, box2):
    """
    Calculates the union of two boxes given their corner coordinates
    
    :param: box1 and box2 specified as for area_box
    :return: union of two boxes in number of pixels
    """
    value = area_box(box1) + area_box(box2) - intersection_boxes(box1, box2)
    return value


def box_iou(box1, box2):
    """
    Calculates the iou of two boxes given their extreme corners coordinates
    
    :param: box1, box2
    :return: intersection over union of the two boxes
    """
    numerator = intersection_boxes(box1, box2)
    denominator = union_boxes(box1, box2)
    return numerator / denominator


def box_ior(box1, box2):
    """
    Calculates the intersection over reference between two boxes (reference box being the second one)

    """
    numerator = intersection_boxes(box1, box2)
    denominator = area_box(box2)
    return numerator / denominator

def median_heuristic(matrix_proba):
    pairwise_dist = squareform(pdist(matrix_proba))
    median_heuristic = np.median(pairwise_dist)
    return median_heuristic



def compute_skeleton(img, axes=None):
    """
    Computes skeleton using skimage.morphology.skeletonize

    Supported input dimensions are:
        img=(batch, channel, *spatial)
        img=(batch, *spatial)
        img=(*spatial)
    where spatial can be a 2D or 3D image.

    Note that, if given, this function iterates over batch and
    channel dimensions, which is very inefficient.
    """
    ndimensions = len(img.shape)
    if ndimensions < 2 or ndimensions > 5:
        raise ValueError(
            f"Image dimensions should be in range [2, 5], got {ndimensions}"
        )
    # Do we have to deal with batch/channel dimensions?
    if axes is None:  axes = [0]
    iter_ix = [i for i in range(0, axes[0])]
    if not iter_ix:
        # No
        return skeletonize(img)
    else:
        # Yes
        skeleton = np.zeros_like(img)
        if len(iter_ix) == 1:
            # Data has channel or batch dimension
            for i in img.shape[0]:
                skeleton[i, ...] = skeletonize(img[i, ...])
        else:
            # Data has both channel and batch dimension
            for i0 in range(img.shape[0]):
                for i1 in range(img.shape[1]):
                    skeleton[i0, i1, ...] = skeletonize(img[i0, i1, ...])
        return skeleton

def compute_box(img):
    indices = np.asarray(np.where(img>0)).T
    min_corner = np.min(indices,0)
    max_corner = np.max(indices, 0)
    box_final = np.concatenate([min_corner, max_corner])
    return box_final

def compute_center_of_mass(img):
    """
    Computes center of mass using scipy.ndimage

    :param: img as multidimensional array
    """
    if img.sum() == 0:
        return -1
    return ndimage.center_of_mass(img)


def distance_transform_edt(img, sampling=None):
    """Computes Euclidean distance transform using ndimage
    """
    return ndimage.distance_transform_edt(
            img, sampling=sampling, return_indices=False
        )

def max_x_at_y_more(x, y, cut_off):
    """Gets max of elements in x where elements 
    in y are geq to a cut off value
    """
    x = np.asarray(x)
    y = np.asarray(y)    
    ix = np.where(y >= cut_off)
    return np.max(x[ix])

def max_x_at_y_less(x, y, cut_off):
    """Gets max of elements in x where elements 
    in y are leq to a cut off value
    """
    x = np.asarray(x)
    y = np.asarray(y)    
    ix = np.where(y <= cut_off)
    return np.max(x[ix])

def min_x_at_y_less(x, y, cut_off):
    """Gets min of elements in x where elements 
    in y are leq to a cut off value

    :param:
    :return: minimum of x such as y is <= cutoff
    """
    x = np.asarray(x)
    y = np.asarray(y)    
    ix = np.where(y <= cut_off)
    return np.min(x[ix])

def min_x_at_y_more(x,y,cut_off):
    """Gets min of elements in x where elements in 
    y are greater than cutoff value
    
    :param: x, y, cutoff
    :return: min of x where y >= cut_off
    """
    x = np.asarray(x)
    y = np.asarray(y)    
    ix = np.where(y >= cut_off)
    return np.min(x[ix])

def one_hot_encode(img, n_classes):
    """One-hot encodes categorical image
    """
    return np.eye(n_classes)[img]

def to_string_count(measures_count, counting_dict, fmt="{:.4f}"):
    result_str = ""
    # list_space = ['com_ref', 'com_pred', 'list_labels']
    for key in measures_count:
        if len(counting_dict[key]) == 2:
            result = counting_dict[key][0]()
        else:
            result = counting_dict[key][0](counting_dict[key][2])
        result_str += (
            ",".join(fmt.format(x) for x in result)
            if isinstance(result, tuple)
            else fmt.format(result)
        )
        result_str += ","
    return result_str[:-1]  # trim the last comma


def to_string_dist(measures_dist, distance_dict, fmt="{:.4f}"):
    result_str = ""
    # list_space = ['com_ref', 'com_pred', 'list_labels']
    for key in measures_dist:
        if len(distance_dict[key]) == 2:
            result = distance_dict[key][0]()
        else:
            result = distance_dict[key][0](distance_dict[key][2])
        result_str += (
            ",".join(fmt.format(x) for x in result)
            if isinstance(result, tuple)
            else fmt.format(result)
        )
        result_str += ","
    return result_str[:-1]  # trim the last comma


def to_string_mt(measures_mthresh, multi_thresholds_dict, fmt="{:.4f}"):
    result_str = ""
    # list_space = ['com_ref', 'com_pred', 'list_labels']
    for key in measures_mthresh:
        if len(multi_thresholds_dict[key]) == 2:
            result = multi_thresholds_dict[key][0]()
        else:
            result = multi_thresholds_dict[key][0](
                multi_thresholds_dict[key][2]
            )
        result_str += (
            ",".join(fmt.format(x) for x in result)
            if isinstance(result, tuple)
            else fmt.format(result)
        )
        result_str += ","
    return result_str[:-1]  # trim the last comma

    
def to_dict_meas_(measures, measures_dict, fmt="{:.4f}"):
    """Given the selected metrics provides a dictionary 
    with relevant metrics"""
    result_dict = {}
    # list_space = ['com_ref', 'com_pred', 'list_labels']
    for key in measures:
        if len(measures_dict[key]) == 2:
            result = measures_dict[key][0]()
        else:
            result = measures_dict[key][0](measures_dict[key][2])
        result_dict[key] = fmt.format(result)
    return result_dict  # trim the last comma



def trapezoidal_integration(x, fx):
    """Trapezoidal integration

    Reference
       https://en.wikipedia.org/w/index.php?title=Trapezoidal_rule&oldid=1104074899#Non-uniform_grid
    """
    return np.sum((fx[:-1] + fx[1:])/2 * (x[1:] - x[:-1]))