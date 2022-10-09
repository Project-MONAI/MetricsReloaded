"""Various utility functions and classes
"""
from functools import partial

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


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
        list_values = np.unique(labels)
        for f in list_values:
            if f > 0:
                tmp_lab = np.where(
                    labels == f, np.ones_like(labels), np.zeros_like(labels)
                )
                list_ind_lab.append(tmp_lab)
        return list_ind_lab


@CacheFunctionOutput
def compute_skeleton(img):
    """
    Computes skeleton using skimage.morphology.skeletonize
    """
    return skeletonize(img)


@CacheFunctionOutput
def compute_center_of_mass(img):
    """
    Computes center of mass using scipy.ndimage
    """
    return ndimage.center_of_mass(img)


def distance_transform_edt(img, sampling=None):
    """Computes Euclidean distance transform using ndimage
    """
    return ndimage.distance_transform_edt(
            img, sampling=sampling, return_indices=False
        )

def x_at_y(x, y, cut_off):
    """Gets max of elements in x where elements 
    in y are geq to a cut off value
    """
    x = np.asarray(x)
    y = np.asarray(y)    
    ix = np.where(y >= cut_off)
    return np.max(x[ix])

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
