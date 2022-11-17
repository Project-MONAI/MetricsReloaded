# Copyright (c) Carole Sudre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pairwise measures - :mod:`MetricsReloaded.metrics.pairwise_measures`
====================================================================

This module provides classes for calculating :ref:`binary
<binary>` and :ref:`multiclass <multiclass>` pairwise measures.

.. _binary:

Calculating binary pairwise measures
------------------------------------

.. autoclass:: BinaryPairwiseMeasures
    :members:

.. _multiclass:

Calculating multiclass pairwise measures
----------------------------------------

.. autoclass:: MultiClassPairwiseMeasures
    :members:
"""


from __future__ import absolute_import, print_function
import warnings
import numpy as np
from scipy import ndimage
from functools import partial
from skimage.morphology import skeletonize
from MetricsReloaded.utility.utils import one_hot_encode, compute_center_of_mass, compute_skeleton, CacheFunctionOutput, MorphologyOps

# from assignment_localization import AssignmentMapping
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.optimize import linear_sum_assignment as lsa


__all__ = [
    'MultiClassPairwiseMeasures',
    'BinaryPairwiseMeasures',
]

# class CacheFunctionOutput(object):
#     """
#     this provides a decorator to cache function outputs
#     to avoid repeating some heavy function computations
#     """

#     def __init__(self, func):
#         self.func = func

#     def __get__(self, obj, _=None):
#         if obj is None:
#             return self
#         return partial(self, obj)  # to remember func as self.func

#     def __call__(self, *args, **kw):
#         obj = args[0]
#         try:
#             cache = obj.__cache
#         except AttributeError:
#             cache = obj.__cache = {}
#         key = (self.func, args[1:], frozenset(kw.items()))
#         try:
#             value = cache[key]
#         except KeyError:
#             value = cache[key] = self.func(*args, **kw)
#         return value





# class MorphologyOps(object):
#     """
#     Class that performs the morphological operations needed to get notably
#     connected component. To be used in the evaluation
#     """

#     def __init__(self, binary_img, neigh):
#         self.binary_map = np.asarray(binary_img, dtype=np.int8)
#         self.neigh = neigh

#     def border_map(self):
#         eroded = ndimage.binary_erosion(self.binary_map)
#         border = self.binary_map - eroded
#         return border

#     def border_map2(self):
#         """
#         Creates the border for a 3D image
#         :return:
#         """
#         west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
#         east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
#         north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
#         south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
#         top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
#         bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
#         cumulative = west + east + north + south + top + bottom
#         border = ((cumulative < 6) * self.binary_map) == 1
#         return border

#     def foreground_component(self):
#         return ndimage.label(self.binary_map)

#     @CacheFunctionOutput
#     def list_foreground_component(self):
#         labels, _ = self.foreground_component()
#         list_ind_lab = []
#         list_vol_lab = []
#         list_com_lab = []
#         list_values = np.unique(labels)
#         for f in list_values:
#             if f > 0:
#                 tmp_lab = np.where(
#                     labels == f, np.ones_like(labels), np.zeros_like(labels)
#                 )
#                 list_ind_lab.append(tmp_lab)
#                 list_vol_lab.append(np.sum(tmp_lab))
#                 list_com_lab.append(np.mean(np.asarray(np.where(tmp_lab==1)).T,0))
#         return list_ind_lab, list_vol_lab, list_com_lab

    
    



class MultiClassPairwiseMeasures(object):
    """Class dealing with measures of direct multi-class such as MCC, Cohen's kappa or balanced accuracy"""

    def __init__(self, pred, ref, list_values, measures=[], dict_args={}):
        self.pred = np.asarray(pred, dtype=np.int32)
        self.ref = np.asarray(ref, dtype=np.int32)
        self.dict_args = dict_args
        self.list_values = list_values
        self.measures = measures
        self.measures_dict = {
            "mcc": (self.matthews_correlation_coefficient, "MCC"),
            "wck": (self.weighted_cohens_kappa, "WCK"),
            "balanced_accuracy": (self.balanced_accuracy, "BAcc"),
            "expected_cost": (self.expected_cost, "EC")
        }

    def expected_cost(self):
        cm = self.confusion_matrix()
        priors = np.sum(cm,0)/np.sum(cm)
        #print(priors,cm)
        numb_perc = np.sum(cm,0)
        rmatrix = cm / numb_perc
        prior_matrix = np.tile(priors,[cm.shape[0],1])
        priorbased_weights = 1.0/(cm.shape[1] * prior_matrix)
        for c in range(cm.shape[0]):
            priorbased_weights[c,c] = 0
        if 'ec_costs' in self.dict_args.keys():
            weights = self.dict_args['ec_costs']
        else:
            weights = priorbased_weights
        print(weights, prior_matrix, rmatrix)
        ec = np.sum(prior_matrix * weights * rmatrix)
        return ec

    def best_naive_ec(self):
        cm = self.confusion_matrix()
        priors = np.sum(cm, 0)/np.sum(cm)
        prior_matrix = np.tile(priors,[cm.shape[0],1])
        priorbased_weights = 1/(cm.shape[1] * prior_matrix)
        for c in range(cm.shape[0]):
            priorbased_weights[c,c] = 0

        if 'ec_costs' in self.dict_args.keys():
            weights = self.dict_args['ec_costs']
        else:
            weights = priorbased_weights
        total_cost = np.sum(weights * prior_matrix,1)
        return np.min(total_cost)

    def normalised_expected_cost(self):
        naive_cost = self.best_naive_ec()
        #print(naive_cost)
        ec = self.expected_cost()
        print(ec, naive_cost)
        return ec / naive_cost

    def matthews_correlation_coefficient(self):
        one_hot_pred = one_hot_encode(self.pred, len(self.list_values))
        one_hot_ref = one_hot_encode(self.ref, len(self.list_values))
        cov_pred = 0
        cov_ref = 0
        cov_pr = 0
        for f in range(len(self.list_values)):
            cov_pred += np.cov(one_hot_pred[:, f], one_hot_pred[:, f])[0, 1]
            cov_ref += np.cov(one_hot_ref[:, f], one_hot_ref[:, f])[0, 1]
            cov_pr += np.cov(one_hot_pred[:, f], one_hot_ref[:, f])[0, 1]
        print(cov_pred, cov_ref, cov_pr)
        numerator = cov_pr
        denominator = np.sqrt(cov_pred * cov_ref)
        return numerator / denominator

    def chance_agreement_probability(self):
        """Determines the probability of agreeing by chance given two classifications.
        To be used for CK calculation"""
        chance = 0
        for f in self.list_values:
            prob_pred = len(np.where(self.pred == f)[0]) / np.size(self.pred)
            prob_ref = len(np.where(self.ref == f)[0]) / np.size(self.ref)
            chance += prob_pred * prob_ref
        return chance

    def confusion_matrix(self):
        """Provides the confusion matrix Prediction in rows, Reference in columns"""
        one_hot_pred = one_hot_encode(self.pred, len(self.list_values))
        one_hot_ref = one_hot_encode(self.ref, len(self.list_values))
        confusion_matrix = np.matmul(one_hot_pred.T, one_hot_ref)
        return confusion_matrix

    def one_hot_pred(self):
        return np.eye(np.max(self.list_values) + 1)[self.pred]

    def one_hot_ref(self):
        return np.eye(np.max(self.list_values) + 1)[self.ref]

    def balanced_accuracy(self):
        """Calculation of balanced accuracy as average of correctly classified
        by reference class across all classes"""
        cm = self.confusion_matrix()
        col_sum = np.sum(cm, 0)
        numerator = np.sum(np.diag(cm) / col_sum)
        denominator = len(self.list_values)
        return numerator / denominator

    def expectation_matrix(self):
        """Determination of the expectation matrix to be used for CK derivation"""
        one_hot_pred = one_hot_encode(self.pred, len(self.list_values))
        one_hot_ref = one_hot_encode(self.ref, len(self.list_values))
        pred_numb = np.sum(one_hot_pred, 0)
        ref_numb = np.sum(one_hot_ref, 0)
        print(pred_numb.shape, ref_numb.shape)
        return (
            np.matmul(np.reshape(pred_numb, [-1, 1]), np.reshape(ref_numb, [1, -1]))
            / np.shape(one_hot_pred)[0]
        )

    def weighted_cohens_kappa(self):
        """Derivation of weighted cohen's kappa. The weight matrix is set to 1-ID(len(list_values))
        - cost of 1 for each error type if no weight provided"""
        cm = self.confusion_matrix()
        exp = self.expectation_matrix()
        if "weights" in self.dict_args.keys():
            weights = self.dict_args["weights"]
        else:
            weights = np.ones([len(self.list_values), len(self.list_values)]) - np.eye(
                len(self.list_values)
            )
        numerator = np.sum(weights * cm)
        denominator = np.sum(weights * exp)
        print(numerator, denominator, cm, exp)
        return 1 - numerator / denominator

    def to_dict_meas(self, fmt="{:.4f}"):
        """Given the selected metrics provides a dictionary with relevant metrics"""
        result_dict = {}
        # list_space = ['com_ref', 'com_pred', 'list_labels']
        for key in self.measures:
            if len(self.measures_dict[key]) == 2:
                result = self.measures_dict[key][0]()
            else:
                result = self.measures_dict[key][0](self.measures_dict[key][2])
            result_dict[key] = fmt.format(result)
        return result_dict  # trim the last comma


class BinaryPairwiseMeasures(object):
    def __init__(
        self,
        pred,
        ref,
        measures=[],
        num_neighbors=8,
        pixdim=None,
        empty=False,
        dict_args={},
    ):

        self.measures_dict = {
            "numb_ref": (self.n_pos_ref, "NumbRef"),
            "numb_pred": (self.n_pos_pred, "NumbPred"),
            "numb_tp": (self.n_intersection, "NumbTP"),
            "numb_fp":(self.fp, "NumbFP"),
            "numb_fn": (self.fn, "NumbFN"),
            "accuracy": (self.accuracy, "Accuracy"),
            "net_benefit":(self.net_benefit_treated, "NB"),
            "expected_cost": (self.normalised_expected_cost, "ECn"),
            "balanced_accuracy": (self.balanced_accuracy, "BalAcc"),
            "cohens_kappa": (self.cohens_kappa, "CohensKappa"),
            "lr+": (self.positive_likelihood_ratio, "LR+"),
            "iou": (self.intersection_over_union, "IoU"),
            "fbeta": (self.fbeta, "FBeta"),
            "youden_ind": (self.youden_index, "YoudenInd"),
            "mcc": (self.matthews_correlation_coefficient, "MCC"),
            "centreline_dsc": (self.centreline_dsc, "CentreLineDSC"),
            "assd": (self.measured_average_distance, "ASSD"),
            "boundary_iou": (self.boundary_iou, "BoundaryIoU"),
            "hd": (self.measured_hausdorff_distance, "HD"),
            "hd_perc": (self.measured_hausdorff_distance_perc, "HDPerc"),
            "masd": (self.measured_masd, "MASD"),
            "nsd": (self.normalised_surface_distance, "NSD"),
        }

        self.pred = pred
        self.ref = ref
        self.flag_empty = empty
        self.measures = measures if measures is not None else self.measures_dict
        self.neigh = num_neighbors
        self.pixdim = pixdim
        self.dict_args = dict_args

    def __fp_map(self):
        """
        This function calculates the false positive map
        :return: FP map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        return np.asarray((pred_float - ref_float) > 0.0, dtype=np.float32)

    def __fn_map(self):
        """
        This function calculates the false negative map
        :return: FN map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        return np.asarray((ref_float - pred_float) > 0.0, dtype=np.float32)

    def __tp_map(self):
        """
        This function calculates the true positive map
        :return: TP map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        return np.asarray((ref_float + pred_float) > 1.0, dtype=np.float32)

    def __tn_map(self):
        """
        This function calculates the true negative map
        :return: TN map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        return np.asarray((ref_float + pred_float) < 0.5, dtype=np.float32)

    def __union_map(self):
        """
        This function calculates the union map between prediction and
        reference image
        :return: union map
        """
        return np.asarray((self.ref + self.pred) > 0.5, dtype=np.float32)

    def __intersection_map(self):
        """
        This function calculates the intersection between prediction and
        reference image
        :return: intersection map
        """
        return np.multiply(self.ref, self.pred)

    @CacheFunctionOutput
    def n_pos_ref(self):
        return np.sum(self.ref)

    @CacheFunctionOutput
    def n_neg_ref(self):
        return np.sum(1 - self.ref)

    @CacheFunctionOutput
    def n_pos_pred(self):
        return np.sum(self.pred)

    @CacheFunctionOutput
    def n_neg_pred(self):
        return np.sum(1 - self.pred)

    @CacheFunctionOutput
    def fp(self):
        return np.sum(self.__fp_map())

    @CacheFunctionOutput
    def fn(self):
        return np.sum(self.__fn_map())

    @CacheFunctionOutput
    def tp(self):
        return np.sum(self.__tp_map())

    @CacheFunctionOutput
    def tn(self):
        return np.sum(self.__tn_map())

    @CacheFunctionOutput
    def n_intersection(self):
        return np.sum(self.__intersection_map())

    @CacheFunctionOutput
    def n_union(self):
        return np.sum(self.__union_map())

    def youden_index(self):
        return self.specificity() + self.sensitivity() - 1

    def sensitivity(self):
        if self.n_pos_ref() == 0:
            warnings.warn("reference empty, sensitivity not defined")
            return np.nan
        return self.tp() / self.n_pos_ref()

    def specificity(self):
        if self.n_neg_ref() == 0:
            warnings.warn("reference all positive, specificity not defined")
            return np.nan
        return self.tn() / self.n_neg_ref()

    def balanced_accuracy(self):
        return 0.5 * self.sensitivity() + 0.5 * self.specificity()

    def accuracy(self):
        return (self.tn() + self.tp()) / (self.tn() + self.tp() + self.fn() + self.fp())

    def false_positive_rate(self):
        return self.fp() / self.n_neg_ref()

    def normalised_expected_cost(self):
        prior_background = (self.tn() + self.fp())/(np.size(self.ref))
        prior_foreground = (self.tp() + self.fn())/np.size(self.ref)

        if 'cost_fn' in self.dict_args.keys():
            c_fn = self.dict_args['cost_fn']
        else:
            c_fn = 1.0/(2*prior_foreground)
        if 'cost_fp' in self.dict_args.keys():
            c_fp = self.dict_args['cost_fp']
        else:
            c_fp = 1.0/(2*prior_background)
        prior_background = (self.tn() + self.fp())/(np.size(self.ref))
        prior_foreground = (self.tp() + self.fn())/np.size(self.ref)
        alpha = c_fp * prior_background / (c_fn * prior_foreground)
        print(prior_background, prior_foreground, alpha)
        r_fp = self.fp()/self.n_neg_ref()
        r_fn = self.fn()/self.n_pos_ref()
        print(r_fn, r_fp)
        if alpha >= 1:
            ecn = alpha * r_fp + r_fn
        else:
            ecn = r_fp + 1/alpha * r_fn
        return ecn

    def matthews_correlation_coefficient(self):
        numerator = self.tp() * self.tn() - self.fp() * self.fn()
        denominator = (
            (self.tp() + self.fp())
            * (self.tp() + self.fn())
            * (self.tn() + self.fp())
            * (self.tn() + self.fn())
        )
        return numerator / np.sqrt(denominator)

    def expected_matching_ck(self):
        list_values = np.unique(self.ref)
        p_e = 0
        for val in list_values:
            p_er = np.sum(
                np.where(
                    self.ref == val, np.ones_like(self.ref), np.zeros_like(self.ref)
                )
            ) / np.prod(self.ref.shape)
            p_es = np.sum(
                np.where(
                    self.pred == val, np.ones_like(self.pred), np.zeros_like(self.pred)
                )
            ) / np.prod(self.pred.shape)
            p_e += p_es * p_er
        return p_e

    def cohens_kappa(self):
        p_e = self.expected_matching_ck()
        p_o = self.accuracy()
        numerator = p_o - p_e
        denominator = 1 - p_e
        return numerator / denominator

    def positive_likelihood_ratio(self):
        numerator = self.sensitivity()
        denominator = 1 - self.specificity()
        return numerator / denominator

    def pred_in_ref(self):
        intersection = np.sum(self.pred * self.ref)
        if intersection > 0:
            return 1
        else:
            return 0

    def positive_predictive_values(self):
        if self.n_pos_pred() == 0:
            if self.n_pos_ref() == 0:
                warnings.warn("ref and prediction empty ppv not defined")
                return np.nan
            else:
                warnings.warn("prediction empty, ppv not defined but set to 0")
                return 0
        return self.tp() / (self.tp() + self.fp())

    def recall(self):
        if self.n_pos_ref() == 0:
            warnings.warn("reference is empty, recall not defined")
            return np.nan
        if self.n_pos_pred() == 0:
            warnings.warn(
                "prediction is empty but ref not, recall not defined but set to 0"
            )
            return 0
        return self.tp() / (self.tp() + self.fn())

    def fbeta(self):
        if "beta" in self.dict_args.keys():
            beta = self.dict_args["beta"]
        else:
            beta = 1
        numerator = (
            (1 + np.square(beta)) * self.positive_predictive_values() * self.recall()
        )
        denominator = (
            np.square(beta) * self.positive_predictive_values() + self.recall()
        )
        print(numerator, denominator, self.fn(), self.tp(), self.fp())
        if np.isnan(denominator):
            if self.fp() + self.fn() > 0:
                return 0
            else:
                return 1  # Potentially modify to nan
        elif denominator == 0:
            if self.fp() + self.fn() > 0:
                return 0
            else:
                return 1  # Potentially modify to nan
        else:
            return numerator / denominator

    def net_benefit_treated(self):
        """
        This functions calculates the net benefit treated according to a specified exchange rate

        .. math::

            NB = \dfrac{TP}{N} - \dfrac{FP}{N} \dot ER
        
        where ER relates to the exchange rate. For instance if a suitable exchange rate is to find 
        1 positive case among 10 tested (1TP for 9 FP), the exchange rate would be 1/9
        """
        if 'exchange_rate' in self.dict_args.keys():
            er = self.dict_args['exchange_rate']
        else:
            er = 1
        n = np.size(self.pred)
        tp = self.tp()
        fp = self.fp()
        nb = tp/n - fp/n * er
        return nb

    def negative_predictive_values(self):
        """
        This function calculates the negative predictive value ratio between
        the number of true negatives and the total number of negative elements
        :return:
        """
        if self.tn() + self.fn() == 0:
            if self.n_neg_ref() == 0:

                warnings.warn(
                    "Nothing negative in either pred or ref, NPV not defined and set to nan"
                )
                return np.nan  # Potentially modify to 1
            else:
                warnings.warn(
                    "Nothing negative in pred but should be NPV not defined but set to 0"
                )
                return 0
        return self.tn() / (self.fn() + self.tn())

    def dice_score(self):
        """
        This function returns the dice score coefficient between a reference
        and prediction images
        :return: dice score
        """
        if not 'fbeta' in self.dict_args.keys():
            self.dict_args['fbeta'] = 1
        elif self.dict_args['fbeta'] != 1:
            warnings.warn('Modifying fbeta option to get dice score')
            self.dict_args['fbeta'] = 1
        else:
            print('Already correct value for fbeta option')
        return self.fbeta()

    def fppi(self):
        """
        This function returns the average number of false positives per
         image, assuming that the cases are collated on the last axis of the array
        """
        sum_per_image = np.sum(
            np.reshape(self.__fp_map(), -1, self.ref.shape[-1]), axis=0
        )
        return np.mean(sum_per_image)

    def intersection_over_reference(self):
        """
        This function the intersection over reference ratio
        :return:
        """
        return self.n_intersection() / self.n_pos_ref()

    def intersection_over_union(self):
        """
        This function the intersection over union ratio - Definition of
        jaccard coefficient
        :return:
        """
        return self.n_intersection() / self.n_union()

    def com_dist(self):
        """
        This function calculates the euclidean distance between the centres
        of mass of the reference and prediction.
        :return:
        """
        print("pred sum ", self.n_pos_pred(), "ref_sum ", self.n_pos_ref())
        if self.flag_empty:
            return -1
        else:
            com_ref = compute_center_of_mass(self.ref)
            com_pred = compute_center_of_mass(self.pred)
            
            print(com_ref, com_pred)
            if self.pixdim is not None:
                com_dist = np.sqrt(
                    np.dot(
                        np.square(np.asarray(com_ref) - np.asarray(com_pred)),
                        np.square(self.pixdim),
                    )
                )
            else:
                com_dist = np.sqrt(
                    np.sum(np.square(np.asarray(com_ref) - np.asarray(com_pred)))
                )
            return com_dist

    def com_ref(self):
        """
        This function calculates the centre of mass of the reference
        prediction
        :return:
        """
        return ndimage.center_of_mass(self.ref)

    def com_pred(self):
        """
        This functions provides the centre of mass of the predmented element
        :return: -1 if empty image, centre of mass of prediction otherwise
        """
        if self.flag_empty:
            return -1
        else:
            return ndimage.center_of_mass(self.pred)

    def list_labels(self):
        if self.list_labels is None:
            return ()
        return tuple(np.unique(self.list_labels))

    def vol_diff(self):
        """
        This function calculates the ratio of difference in volume between
        the reference and prediction images.
        :return: vol_diff
        """
        return np.abs(self.n_pos_ref() - self.n_pos_pred()) / self.n_pos_ref()

    @CacheFunctionOutput
    def skeleton_versions(self):
        skeleton_ref = compute_skeleton(self.ref)
        skeleton_pred = compute_skeleton(self.pred)
        return skeleton_ref, skeleton_pred

    def topology_precision(self):
        skeleton_ref, skeleton_pred = self.skeleton_versions()
        numerator = np.sum(skeleton_pred * self.ref)
        denominator = np.sum(skeleton_pred)
        print("top prec ", numerator, denominator)
        return numerator / denominator

    def topology_sensitivity(self):
        skeleton_ref, skeleton_pred = self.skeleton_versions()
        numerator = np.sum(skeleton_ref * self.pred)
        denominator = np.sum(skeleton_ref)
        print("top sens ", numerator, denominator, skeleton_ref, skeleton_pred)
        return numerator / denominator

    def centreline_dsc(self):
        top_prec = self.topology_precision()
        top_sens = self.topology_sensitivity()
        numerator = 2 * top_sens * top_prec
        denominator = top_sens + top_prec
        return numerator / denominator

    def boundary_iou(self):
        """
        This functions determines the boundary iou
        """
        if 'boundary_dist' in self.dict_args.keys():
            distance = self.dict_args['boundary_dist']
        else:
            distance = 1
        border_ref = MorphologyOps(self.ref, self.neigh).border_map()
        distance_border_ref = ndimage.distance_transform_edt(1-border_ref)

        border_pred = MorphologyOps(self.pred, self.neigh).border_map()
        distance_border_pred = ndimage.distance_transform_edt(1-border_pred)

        lim_dbp = np.where(distance_border_pred < distance, np.ones_like(border_pred), np.zeros_like(border_pred))
        lim_dbr = np.where(distance_border_ref < distance, np.ones_like(border_ref), np.zeros_like(border_ref))

        intersect = np.sum(lim_dbp*lim_dbr)
        union = np.sum(np.where(lim_dbp + lim_dbr>0,np.ones_like(border_ref),np.zeros_like(border_pred)))
        print(intersect, union)
        return intersect / union
        # return np.sum(border_ref * border_pred) / (
        #     np.sum(border_ref) + np.sum(border_pred)
        # )

    @CacheFunctionOutput
    def border_distance(self):
        """
        This functions determines the map of distance from the borders of the
        prediction and the reference and the border maps themselves
        :return: distance_border_ref, distance_border_pred, border_ref,
        border_pred
        """
        border_ref = MorphologyOps(self.ref, self.neigh).border_map()
        border_pred = MorphologyOps(self.pred, self.neigh).border_map()
        oppose_ref = 1 - self.ref
        oppose_pred = 1 - self.pred
        distance_ref = ndimage.distance_transform_edt(
            1 - border_ref, sampling=self.pixdim
        )
        distance_pred = ndimage.distance_transform_edt(
            1 - border_pred, sampling=self.pixdim
        )
        distance_border_pred = border_ref * distance_pred
        distance_border_ref = border_pred * distance_ref
        return distance_border_ref, distance_border_pred, border_ref, border_pred

    def normalised_surface_distance(self):
        if "nsd" in self.dict_args.keys():
            tau = self.dict_args["nsd"]
        else:
            tau = 1
        dist_ref, dist_pred, border_ref, border_pred = self.border_distance()
        reg_ref = np.where(
            dist_ref <= tau, np.ones_like(dist_ref), np.zeros_like(dist_ref)
        )
        reg_pred = np.where(
            dist_pred <= tau, np.ones_like(dist_pred), np.zeros_like(dist_pred)
        )
        numerator = np.sum(border_pred * reg_ref) + np.sum(border_ref * reg_pred)
        denominator = np.sum(border_ref) + np.sum(border_pred)
        return numerator / denominator

    def measured_distance(self, perc=95):
        """
        This functions calculates the average symmetric distance and the
        hausdorff distance between a prediction and a reference image
        :return: hausdorff distance and average symmetric distance
        """
        if 'hd_perc' in self.dict_args.keys():
            perc = self.dict_args['hd_perc']
        else:
            perc = perc
        if np.sum(self.pred + self.ref) == 0:
            return 0, 0, 0
        (
            ref_border_dist,
            pred_border_dist,
            ref_border,
            pred_border,
        ) = self.border_distance()
        average_distance = (np.sum(ref_border_dist) + np.sum(pred_border_dist)) / (
            np.sum(pred_border + ref_border)
        )
        masd = 0.5 * (
            np.sum(ref_border_dist) / np.sum(pred_border)
            + np.sum(pred_border_dist) / np.sum(ref_border)
        )
        print(
            np.sum(ref_border_dist) / np.sum(pred_border),
            np.sum(pred_border_dist) / np.sum(ref_border),
            np.sum(pred_border),
            np.sum(ref_border),
            np.sum(pred_border_dist),
            np.sum(ref_border_dist),
        )
        hausdorff_distance = np.max([np.max(ref_border_dist), np.max(pred_border_dist)])
        hausdorff_distance_perc = np.max(
            [
                np.percentile(ref_border_dist[self.ref + self.pred > 0], q=perc),
                np.percentile(pred_border_dist[self.ref + self.pred > 0], q=perc),
            ]
        )
        print(ref_border_dist[self.ref + self.pred > 0], pred_border_dist[self.ref + self.pred > 0])
        print(len(ref_border_dist[self.ref + self.pred > 0]), len(pred_border_dist[self.ref + self.pred > 0]))
        
        return hausdorff_distance, average_distance, hausdorff_distance_perc, masd

    def measured_average_distance(self):
        """
        This function returns only the average distance when calculating the
        distances between prediction and reference
        :return:
        """
        return self.measured_distance()[1]

    def measured_masd(self):
        return self.measured_distance()[3]

    def measured_hausdorff_distance(self):
        """
        This function returns only the hausdorff distance when calculated the
        distances between prediction and reference
        :return:
        """
        return self.measured_distance()[0]

    def measured_hausdorff_distance_perc(self):
        if "hd" in self.dict_args.keys():
            perc = self.dict_args["hd"]
        else:
            perc = 95
        return self.measured_distance(perc)[2]

    def to_dict_meas(self, fmt="{:.4f}"):
        result_dict = {}
        for key in self.measures:
            if len(self.measures_dict[key]) == 2:
                result = self.measures_dict[key][0]()
            else:
                result = self.measures_dict[key][0](self.measures_dict[key][2])
            result_dict[key] = fmt.format(result)
        return result_dict  # trim the last comma

    def to_string_count(self, fmt="{:.4f}"):
        result_str = ""
        for key in self.measures_count:
            if len(self.counting_dict[key]) == 2:
                result = self.counting_dict[key][0]()
            else:
                result = self.counting_dict[key][0](self.counting_dict[key][2])
            result_str += (
                ",".join(fmt.format(x) for x in result)
                if isinstance(result, tuple)
                else fmt.format(result)
            )
            result_str += ","
        return result_str[:-1]  # trim the last comma

    def to_string_dist(self, fmt="{:.4f}"):
        result_str = ""
        
        for key in self.measures_dist:
            if len(self.distance_dict[key]) == 2:
                result = self.distance_dict[key][0]()
            else:
                result = self.distance_dict[key][0](self.distance_dict[key][2])
            result_str += (
                ",".join(fmt.format(x) for x in result)
                if isinstance(result, tuple)
                else fmt.format(result)
            )
            result_str += ","
        return result_str[:-1]  # trim the last comma

    def to_string_mt(self, fmt="{:.4f}"):
        result_str = ""
        
        for key in self.measures_mthresh:
            if len(self.multi_thresholds_dict[key]) == 2:
                result = self.multi_thresholds_dict[key][0]()
            else:
                result = self.multi_thresholds_dict[key][0](
                    self.multi_thresholds_dict[key][2]
                )
            result_str += (
                ",".join(fmt.format(x) for x in result)
                if isinstance(result, tuple)
                else fmt.format(result)
            )
            result_str += ","
        return result_str[:-1]  # trim the last comma
