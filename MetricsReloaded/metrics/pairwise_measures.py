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
from MetricsReloaded.utility.utils import (
    one_hot_encode,
    compute_center_of_mass,
    compute_skeleton,
    CacheFunctionOutput,
    MorphologyOps,
)

# from assignment_localization import AssignmentMapping
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.optimize import linear_sum_assignment as lsa

__all__ = [
    "MultiClassPairwiseMeasures",
    "BinaryPairwiseMeasures",
]


class MultiClassPairwiseMeasures(object):
    """

    Class dealing with measures of direct multi-class such as MCC, Cohen's kappa, Expected cost
    or balanced accuracy


    """

    def __init__(self, pred, ref, list_values,
        measures=[], dict_args={}, smooth_dr=0, axis=None, is_onehot=False):
        self.pred = np.asarray(pred, dtype=np.int32)
        self.ref = np.asarray(ref, dtype=np.int32)
        self.dict_args = dict_args
        self.list_values = list_values
        self.measures = measures
        self.measures_dict = {
            "mcc": (self.matthews_correlation_coefficient, "MCC"),
            "wck": (self.weighted_cohens_kappa, "WCK"),
            "balanced_accuracy": (self.balanced_accuracy, "BAcc"),
            "expected_cost": (self.expected_cost, "EC"),
        }
        self.n_classes = len(self.list_values)

        self.metrics = {
            "Balanced Accuracy": self.balanced_accuracy,
            "Weighted Cohens Kappa": self.weighted_cohens_kappa,
            "Matthews Correlation Coefficient": self.matthews_correlation_coefficient,
            "Expected Cost": self.expected_cost,
            "Normalised Expected Cost": self.normalised_expected_cost,
        }
        self.smooth_dr = smooth_dr
        self.axis = axis
        if self.axis == None:
            self.axis = (0, 1)
        self.is_onehot = is_onehot

    def confusion_matrix(self, return_onehot=False):
        """
        Provides the confusion matrix Prediction in rows, Reference in columns

        :return: confusion_matrix
        """
        if self.is_onehot:
            one_hot_pred = self.pred
            one_hot_ref = self.ref
        else:
            one_hot_pred = one_hot_encode(self.pred, self.n_classes)
            one_hot_ref = one_hot_encode(self.ref, self.n_classes)
        confusion_matrix = np.matmul(np.swapaxes(one_hot_pred, -1, -2), one_hot_ref)
        if return_onehot:
            return confusion_matrix, one_hot_pred, one_hot_ref
        else:
            return confusion_matrix

    def expectation_matrix(self, one_hot_pred=None, one_hot_ref=None):
        """
        Determination of the expectation matrix to be used for CK derivation

        :return: expectation_matrix
        """
        if one_hot_pred is None:
            one_hot_pred = one_hot_encode(self.pred, self.n_classes)
        if one_hot_ref is None:
            one_hot_ref = one_hot_encode(self.ref, self.n_classes)
        pred_numb = np.sum(one_hot_pred, axis=len(one_hot_pred.shape) - 2)
        ref_numb = np.sum(one_hot_ref, axis=len(one_hot_pred.shape) - 2)
        n = one_hot_pred.shape[-2]
        # print(pred_numb.shape, ref_numb.shape)
        out = np.matmul(np.expand_dims(pred_numb, -1), np.expand_dims(ref_numb, -2)) / n
        return out

    def balanced_accuracy(self):
        """Calculation of balanced accuracy as average of correctly classified
        by reference class across all classes

        .. math::

            BA = \dfrac{\sum_{k=1}^{K} \dfrac{TP_k}{TP_k+FN_k}}{K}

        :return: balanced_accuracy
        """
        cm = self.confusion_matrix()
        col_sum = np.sum(cm, axis=len(cm.shape) - 2)
        numerator = np.diagonal(cm, axis1=len(cm.shape) - 2, axis2=len(cm.shape) - 1)
        numerator = numerator/col_sum
        numerator = numerator.sum(-1)
        denominator = self.n_classes
        return numerator / denominator

    def expected_cost(self, normalise=False):
        cm = self.confusion_matrix()
        priors = np.sum(cm, axis=len(cm.shape) - 2, keepdims=True) \
            / np.sum(cm, axis=self.axis, keepdims=True)
        prior_matrix = np.tile(priors, [cm.shape[-2], 1])
        priorbased_weights = 1.0 / (cm.shape[-2] * prior_matrix)
        for c in range(cm.shape[-2]):
            priorbased_weights[..., c, c] = 0
        if "ec_costs" in self.dict_args.keys():
            weights = self.dict_args["ec_costs"]
        else:
            weights = priorbased_weights
        numb_perc = np.sum(cm, axis=len(cm.shape) - 2, keepdims=True)
        rmatrix = cm / numb_perc
        ec = np.sum(prior_matrix * weights * rmatrix, axis=self.axis)
        if normalise:
            # Normalise with best naive expected cost
            bnec = np.sum(weights * prior_matrix, axis=len(cm.shape) - 1)
            bnec = np.min(bnec, axis=-1)
            return ec / bnec
        else:
            return ec

    def normalised_expected_cost(self):
        nec = self.expected_cost(normalise=True)
        # print(nec)
        return nec

    def matthews_correlation_coefficient(self):
        """
        Calculates the multiclass Matthews Correlation Coefficient defined as

        .. math::

            R_k = \dfrac{cov_k(Pred,Ref)}{\sqrt{cov_k(Pred,Pred)*cov_k(Ref,Ref)}}

        with

        .. math::
            cov_k(X,Y) = \dfrac{1}{K}\sum_{k=1}^{K}cov(X_k,Y_k)

        Reference
        Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC and CEN
        Error Measures in MultiClass Prediction
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041882

        :return: Matthews Correlation Coefficient
        """
        if self.is_onehot:
            one_hot_pred = self.pred
            one_hot_ref = self.ref
        else:
            one_hot_pred = one_hot_encode(self.pred, self.n_classes)
            one_hot_ref = one_hot_encode(self.ref, self.n_classes)

        p_bar = np.mean(one_hot_pred, axis=len(one_hot_pred.shape) - 2, keepdims=True)
        r_bar = np.mean(one_hot_ref, axis=len(one_hot_pred.shape) - 2, keepdims=True)

        pp_cov = 1/self.n_classes * np.sum((one_hot_pred - p_bar)**2, axis=self.axis)
        rr_cov = 1/self.n_classes * np.sum((one_hot_ref - r_bar)**2, axis=self.axis)
        pr_cov = 1/self.n_classes * np.sum((one_hot_pred - p_bar) * (one_hot_ref - r_bar), axis=self.axis)

        mcc = pr_cov/np.sqrt(rr_cov * pp_cov)

        return mcc

    def chance_agreement_probability(self):
        """Determines the probability of agreeing by chance given two classifications.
        To be used for CK calculation


        """
        chance = 0
        for f in self.list_values:
            prob_pred = len(np.where(self.pred == f)[0]) / np.size(self.pred)
            prob_ref = len(np.where(self.ref == f)[0]) / np.size(self.ref)
            chance += prob_pred * prob_ref
        return chance

    def weighted_cohens_kappa(self):
        """
        Derivation of weighted cohen's kappa. The weight matrix is set to 1-ID(len(list_values))
        - cost of 1 for each error type if no weight provided

        :return: weighted_cohens_kappa
        """
        cm, one_hot_pred, one_hot_ref = self.confusion_matrix(return_onehot=True)
        exp = self.expectation_matrix(one_hot_pred=one_hot_pred, one_hot_ref=one_hot_ref)
        if "weights" in self.dict_args.keys():
            weights = self.dict_args["weights"]
        else:
            weights = np.ones([self.n_classes, self.n_classes]) \
                - np.eye(self.n_classes)
        if len(cm.shape) == 3:
            # Has batch dimension
            weights = np.tile(weights,(cm.shape[0], 1, 1))
        numerator = np.sum(weights * cm, axis=self.axis)
        denominator = np.sum(weights * exp, axis=self.axis)
        # print(numerator, denominator, cm, exp)
        return 1 - numerator / denominator

    def to_dict_meas(self, fmt="{:.4f}"):
        """Given the selected metrics provides a dictionary with relevant metrics"""
        result_dict = {}
        for key in self.measures:
            result = self.measures_dict[key][0]()
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
        axis=None,
        smooth_dr=0,
    ):

        self.measures_dict = {
            "numb_ref": (self.n_pos_ref, "NumbRef"),
            "numb_pred": (self.n_pos_pred, "NumbPred"),
            "numb_tp": (self.n_intersection, "NumbTP"),
            "numb_fp": (self.fp, "NumbFP"),
            "numb_fn": (self.fn, "NumbFN"),
            "accuracy": (self.accuracy, "Accuracy"),
            "net_benefit": (self.net_benefit_treated, "NB"),
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
        self.flag_empty_pred = False
        self.flag_empty_ref = False
        if np.sum(self.pred) == 0:
            self.flag_empty_pred = True
        if np.sum(self.ref) == 0:
            self.flag_empty_ref = True
        self.measures = measures if measures is not None else self.measures_dict
        self.neigh = num_neighbors
        self.pixdim = pixdim
        self.dict_args = dict_args

        self.metrics = {
            "False Positives": self.fp,
            "False Negatives": self.fn,
            "True Positives": self.tp,
            "True Negatives": self.tn,
            "Youden Index": self.youden_index,
            "Sensitivity": self.sensitivity,
            "Specificity": self.specificity,
            "Balanced Accuracy": self.balanced_accuracy,
            "Accuracy": self.accuracy,
            "False Positive Rate": self.false_positive_rate,
            "Normalised Expected Cost": self.normalised_expected_cost,
            "Matthews Correlation Coefficient": self.matthews_correlation_coefficient,
            "Cohens Kappa": self.cohens_kappa,
            "Positive Likelihood Ratio": self.positive_likelihood_ratio,
            "Prediction Overlaps Reference": self.pred_in_ref,
            "Positive Predictive Value": self.positive_predictive_values,
            "Recall": self.recall,
            "FBeta": self.fbeta,
            "Net Benefit Treated": self.net_benefit_treated,
            "Negative Predictive Values": self.negative_predictive_values,
            "Dice Score": self.dice_score,
            "False Positives Per Image": self.fppi,
            "Intersection Over Reference": self.intersection_over_reference,
            "Intersection Over Union": self.intersection_over_union,
            "Volume Difference": self.vol_diff,
            "Topology Precision": self.topology_precision,
            "Topology Sensitivity": self.topology_sensitivity,
            "Centreline Dice Score": self.centreline_dsc,
            "Boundary IoU": self.boundary_iou,
            "Normalised Surface Distance": self.normalised_surface_distance,
            "Average Symmetric Surface Distance": self.measured_average_distance,
            "Mean Average Surfance Distance": self.measured_masd,
            "Hausdorff Distance": self.measured_hausdorff_distance,
            "xTh Percentile Hausdorff Distance": self.measured_hausdorff_distance_perc,
        }
        self.smooth_dr = smooth_dr
        self.axis = axis
        if self.axis == None:
            self.axis = tuple(range(len(pred.shape)))

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
        """
        Returns the number of elements in ref
        """
        return np.sum(self.ref, axis=self.axis)

    @CacheFunctionOutput
    def n_neg_ref(self):
        """
        Returns the number of negative elements in ref
        """
        return np.sum(1 - self.ref, axis=self.axis)

    @CacheFunctionOutput
    def n_pos_pred(self):
        """
        Returns the number of positive elements in the prediction
        """
        return np.sum(self.pred, axis=self.axis)

    @CacheFunctionOutput
    def n_neg_pred(self):
        """
        Returns the number of negative elements in the prediction
        """
        return np.sum(1 - self.pred, axis=self.axis)

    @CacheFunctionOutput
    def fp(self):
        """
        Calculates the number of FP as sum of elements in FP_map
        """
        return np.sum(self.__fp_map(), axis=self.axis)

    @CacheFunctionOutput
    def fn(self):
        """
        Calculates the number of FN as sum of elements of FN_map
        """
        return np.sum(self.__fn_map(), axis=self.axis)

    @CacheFunctionOutput
    def tp(self):
        """
        Returns the number of true positive (TP) elements
        """
        return np.sum(self.__tp_map(), axis=self.axis)

    @CacheFunctionOutput
    def tn(self):
        """
        Returns the number of True Negative (TN) elements
        """
        return np.sum(self.__tn_map(), axis=self.axis)

    @CacheFunctionOutput
    def n_intersection(self):
        """
        Returns the number of elements in the intersection of reference and prediction (=TP)
        """
        return np.sum(self.__intersection_map())

    @CacheFunctionOutput
    def n_union(self):
        """
        Returns the number of elements in the union of reference and prediction

        .. math::

            U = {\vert} Pred {\vert} + {\vert} Ref {\vert} - TP

        """
        return np.sum(self.__union_map(), axis=self.axis)

    @CacheFunctionOutput
    def skeleton_versions(self, return_pred=True, return_ref=True):
        """
        Creates the skeletonised version of both reference and prediction

        :return: skeleton_ref, skeleton_pred
        """
        skeleton_ref = None
        if return_ref:
            skeleton_ref = compute_skeleton(self.ref, axes=self.axis)
        skeleton_pred = None
        if return_pred:
            skeleton_pred = compute_skeleton(self.pred, axes=self.axis)
        return skeleton_ref, skeleton_pred

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
        distance_ref = ndimage.distance_transform_edt(
            1 - border_ref, sampling=self.pixdim
        )
        distance_pred = ndimage.distance_transform_edt(
            1 - border_pred, sampling=self.pixdim
        )
        distance_border_pred = border_ref * distance_pred
        distance_border_ref = border_pred * distance_ref
        return distance_border_ref, distance_border_pred, border_ref, border_pred

    def youden_index(self):
        """
        Calculates the Youden Index (YI) defined as:

        .. math::

            YI = Specificity + Sensitivity - 1

        """
        return self.specificity() + self.sensitivity() - 1

    def sensitivity(self):
        """
        Calculates the sensitivity defined as

        .. math::

            Sens = \dfrac{TP}{\sharp Ref}

        This measure is not defined for empty reference. Will raise a warning and return a nan value

        :return: sensitivity
        """
        if self.smooth_dr == 0 and self.n_pos_ref() == 0:
            warnings.warn("reference empty, sensitivity not defined")
            return np.nan
        return self.tp() / (self.n_pos_ref() + self.smooth_dr)

    def specificity(self):
        """
        Calculates the specificity defined as

        .. math::

            Spec = \dfrac{TN}{\sharp {1-Ref}}

        This measure is not defined when there is no reference negative. This will
        raise a warning and return a nan

        :return: specificity
        """
        if self.smooth_dr == 0 and self.n_neg_ref() == 0:
            warnings.warn("reference all positive, specificity not defined")
            return np.nan
        return self.tn() / (self.n_neg_ref() + self.smooth_dr)

    def balanced_accuracy(self):
        """
        Calculates and returns the balanced accuracy defined for the
        binary case as the average between sensitivity and specificity

        :return: balanced accuracy
        """
        return 0.5 * self.sensitivity() + 0.5 * self.specificity()

    def accuracy(self):
        """
        Calculate and returns the accuracy defined as

        .. math::

            Acc = \dfrac{TN+TP}{TN+TP+FN+FP}

        :return: accuracy
        """
        return (self.tn() + self.tp()) / (self.tn() + self.tp() + self.fn() + self.fp())

    def false_positive_rate(self):
        """
        Calculates and returns the false positive rate defined as

        .. math::

            FPR = \dfrac{FP}{\sharp \bar{Ref}}

        :return: false positive rate
        """
        return self.fp() / self.n_neg_ref()

    def normalised_expected_cost(self):
        prior_background = (self.tn() + self.fp()) / (np.size(self.ref))
        prior_foreground = (self.tp() + self.fn()) / np.size(self.ref)

        if "cost_fn" in self.dict_args.keys():
            c_fn = self.dict_args["cost_fn"]
        else:
            c_fn = 1.0 / (2 * prior_foreground)
        if "cost_fp" in self.dict_args.keys():
            c_fp = self.dict_args["cost_fp"]
        else:
            c_fp = 1.0 / (2 * prior_background)
        prior_background = (self.tn() + self.fp()) / (np.size(self.ref))
        prior_foreground = (self.tp() + self.fn()) / np.size(self.ref)
        alpha = c_fp * prior_background / (c_fn * prior_foreground)
        # print(prior_background, prior_foreground, alpha)
        r_fp = self.fp() / self.n_neg_ref()
        r_fn = self.fn() / self.n_pos_ref()
        # print(r_fn, r_fp)
        msk = alpha >= 1
        ecn = np.zeros_like(alpha)
        ecn[msk] = alpha[msk] * r_fp[msk] + r_fn[msk]
        ecn[~msk] = r_fp[~msk] + 1 / alpha[~msk] * r_fn[~msk]
        return ecn

    def matthews_correlation_coefficient(self):
        """
        Calculates and returns the MCC for the binary case

        .. math::

            MCC = \dfrac{TP * TN - FP * FN}{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}

        :return: MCC
        """
        numerator = self.tp() * self.tn() - self.fp() * self.fn()
        denominator = (
            (self.tp() + self.fp())
            * (self.tp() + self.fn())
            * (self.tn() + self.fp())
            * (self.tn() + self.fn())
        )
        return numerator / np.sqrt(denominator)

    def cohens_kappa(self):
        """
        Calculates and return the Cohen's kappa score defined as

        .. math::

            CK = \dfrac{p_o - p_e}{1-p_e}

        where

        :math:`p_e = ` expected chance matching and :math:`p_o = `observed accuracy

        :return: CK
        """
        def expected_matching():
            list_values = np.unique(self.ref)
            p_e = 0
            n = 1
            for i in self.axis:  n *= self.pred.shape[i]
            for val in list_values:
                p_er = np.sum(
                    np.where(
                        self.ref == val, np.ones_like(self.ref), np.zeros_like(self.ref)
                    ), axis=self.axis,
                ) / n
                p_es = np.sum(
                    np.where(
                        self.pred == val, np.ones_like(self.pred), np.zeros_like(self.pred)
                    ), axis=self.axis,
                ) / n
                p_e += p_es * p_er
            return p_e

        p_e = expected_matching()
        p_o = self.accuracy()
        numerator = p_o - p_e
        denominator = 1 - p_e
        return numerator / denominator

    def positive_likelihood_ratio(self):
        """
        Calculates the positive likelihood ratio

        .. math::

            LR+ = \dfrac{Sensitivity}{1-Specificity}

        :return: LR+
        """
        numerator = self.sensitivity()
        denominator = 1 - self.specificity()
        return numerator / denominator

    def pred_in_ref(self):
        """
        Determines if prediction and reference overlap on at least one voxel.

        :return: 1 if true, 0 otherwise
        """
        intersection = np.sum(self.pred * self.ref, self.axis)
        return np.where(intersection > 0, 1, 0)

    def positive_predictive_values(self):
        """
        Calculates the positive predictive value

        .. math::

            PPV = \dfrac{TP}{TP+FP}

        Not defined when no positives in the prediction - returns nan if both
        reference and prediction empty. Returns 0 if only prediction empty

        :return: PPV
        """
        if self.smooth_dr == 0 and self.flag_empty_pred:
            if self.flag_empty_ref:
                warnings.warn("ref and prediction empty ppv not defined")
                return np.nan
            else:
                warnings.warn("prediction empty, ppv not defined but set to 0")
                return 0
        return self.tp() / (self.tp() + self.fp() + self.smooth_dr)

    def recall(self):
        """
        Calculates and returns recall = sensitivity

        :return: Recall = Sensitivity
        """
        if self.smooth_dr == 0 and self.n_pos_ref() == 0:
            warnings.warn("reference is empty, recall not defined")
            return np.nan
        if self.smooth_dr == 0 and self.n_pos_pred() == 0:
            warnings.warn(
                "prediction is empty but ref not, recall not defined but set to 0"
            )
            return 0
        return self.sensitivity()

    def fbeta(self):
        """
        Calculates FBeta score defined as

        .. math::

            F_{\\beta} = (1+{\\beta}^2) \dfrac{Precision  * Recall}{{\\beta}^2 * Precision + recall}

        When :math:`{\\beta}=1` it corresponds to the dice score. The :math:`{\\beta}` parameter is
        set up in the class dictionary of options

        :return: fbeta value

        """
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
        # print(numerator, denominator, self.fn(), self.tp(), self.fp())
        if self.smooth_dr == 0 and np.isnan(denominator):
            if self.fp() + self.fn() > 0:
                return 0
            else:
                return 1  # Potentially modify to nan
        elif self.smooth_dr == 0 and denominator == 0:
            if self.fp() + self.fn() > 0:
                return 0
            else:
                return 1  # Potentially modify to nan
        else:
            return numerator / (denominator + self.smooth_dr)

    def net_benefit_treated(self):
        """
        This functions calculates the net benefit treated according to a specified exchange rate

        .. math::

            NB = \dfrac{TP}{N} - \dfrac{FP}{N} * ER

        where ER relates to the exchange rate. For instance if a suitable exchange rate is to find
        1 positive case among 10 tested (1TP for 9 FP), the exchange rate would be 1/9

        :return: NB
        """
        if "exchange_rate" in self.dict_args.keys():
            er = self.dict_args["exchange_rate"]
        else:
            er = 1
        n = 1
        for i in self.axis:  n *= self.pred.shape[i]
        tp = self.tp()
        fp = self.fp()
        nb = tp / n - fp / n * er
        return nb

    def negative_predictive_values(self):
        """
        This function calculates the negative predictive value ratio between
        the number of true negatives and the total number of negative elements

        :return: NPV
        """
        if self.smooth_dr == 0 and self.tn() + self.fn() == 0:
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
        return self.tn() / (self.fn() + self.tn() + self.smooth_dr)

    def dice_score(self):
        """
        This function returns the dice score coefficient between a reference
        and prediction images

        :return: dice score
        """
        if not "fbeta" in self.dict_args.keys():
            self.dict_args["fbeta"] = 1
        elif self.dict_args["fbeta"] != 1:
            warnings.warn("Modifying fbeta option to get dice score")
            self.dict_args["fbeta"] = 1
        # else:
        #     print("Already correct value for fbeta option")
        return self.fbeta()

    def fppi(self):
        """
        This function returns the average number of false positives per image
        """
        return self.__fp_map().mean(axis=self.axis)

    def intersection_over_reference(self):
        """
        This function calculates the ratio of the intersection of prediction and
        reference over reference.

        :return: IoR
        """
        if self.smooth_dr == 0 and self.flag_empty_ref:
            warnings.warn("Empty reference")
            return np.nan
        return self.n_intersection() / (self.n_pos_ref() + self.smooth_dr)

    def intersection_over_union(self):
        """
        This function calculates the intersection of prediction and
        reference over union - This is also the definition of
        jaccard coefficient

        :return: IoU
        """
        if self.smooth_dr == 0 and self.flag_empty_pred and self.flag_empty_ref:
            warnings.warn("Both reference and prediction are empty")
            return np.nan
        return self.n_intersection() / (self.n_union() + self.smooth_dr)

    def com_dist(self):
        """
        This function calculates the euclidean distance between the centres
        of mass of the reference and prediction.

        :return: Euclidean distance between centre of mass when reference and prediction not empty
        -1 otherwise
        """
        # print("pred sum ", self.n_pos_pred(), "ref_sum ", self.n_pos_ref())
        if self.flag_empty_pred or self.flag_empty_ref:
            return -1
        else:
            com_ref = compute_center_of_mass(self.ref)
            com_pred = compute_center_of_mass(self.pred)

            # print(com_ref, com_pred)
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

    def vol_diff(self):
        """
        This function calculates the ratio of difference in volume between
        the reference and prediction images.

        :return: vol_diff
        """
        return np.abs(self.n_pos_ref() - self.n_pos_pred()) / self.n_pos_ref()

    def topology_precision(self):
        """
        Calculates topology precision defined as

        .. math::

            Prec_{Top} = \dfrac{|S_{Pred} \cap Ref|}{|S_{Pred}|}

        with :math:`S_{Pred}` the skeleton of Pred

        :return: topology_precision
        """
        _, skeleton_pred = self.skeleton_versions(return_ref=False)
        numerator = np.sum(skeleton_pred * self.ref, axis=self.axis)
        denominator = np.sum(skeleton_pred, axis=self.axis)
        # print("top prec ", numerator, denominator)
        return numerator / denominator

    def topology_sensitivity(self):
        """
        Calculates the topology sensitivity defined as

        .. math::

            Sens_{Top} = \dfrac{|S_{Ref} \cap Pred|}{|S_{Ref}|}

        with :math:`S_{Ref}` the skeleton of Ref

        :return: topology_sensitivity
        """
        skeleton_ref, _ = self.skeleton_versions(return_pred=False)
        numerator = np.sum(skeleton_ref * self.pred, axis=self.axis)
        denominator = np.sum(skeleton_ref, axis=self.axis)
        # print("top sens ", numerator, denominator, skeleton_ref, skeleton_pred)
        return numerator / denominator

    def centreline_dsc(self):
        """
        Calculates the centre line dice score defined as

        .. math::

            cDSC = 2\dfrac{Sens_{Top} * Prec_{Top}}{Sens_{Top} + Prec_{Top}}

        :return: cDSC
        """
        top_prec = self.topology_precision()
        top_sens = self.topology_sensitivity()
        numerator = 2 * top_sens * top_prec
        denominator = top_sens + top_prec
        return numerator / denominator

    def boundary_iou(self):
        """
        This functions determines the boundary iou

        :return: boundary_iou
        """
        if "boundary_dist" in self.dict_args.keys():
            distance = self.dict_args["boundary_dist"]
        else:
            distance = 1
        border_ref = MorphologyOps(self.ref, self.neigh).border_map()
        distance_border_ref = ndimage.distance_transform_edt(1 - border_ref)

        border_pred = MorphologyOps(self.pred, self.neigh).border_map()
        distance_border_pred = ndimage.distance_transform_edt(1 - border_pred)

        lim_dbp = np.where(
            distance_border_pred < distance,
            np.ones_like(border_pred),
            np.zeros_like(border_pred),
        )
        lim_dbr = np.where(
            distance_border_ref < distance,
            np.ones_like(border_ref),
            np.zeros_like(border_ref),
        )

        intersect = np.sum(lim_dbp * lim_dbr, axis=self.axis)
        union = np.sum(
            np.where(
                lim_dbp + lim_dbr > 0,
                np.ones_like(border_ref),
                np.zeros_like(border_pred),
            ), axis=self.axis
        )
        # print(intersect, union)
        return intersect / union
        # return np.sum(border_ref * border_pred) / (
        #     np.sum(border_ref) + np.sum(border_pred)
        # )

    def normalised_surface_distance(self):
        """
        Calculates the normalised surface distance (NSD) between prediction and reference
        using the distance parameter :math:`{\\tau}`

        :return: NSD
        """
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
        numerator = np.sum(border_pred * reg_ref, axis=self.axis) + np.sum(border_ref * reg_pred, axis=self.axis)
        denominator = np.sum(border_ref, axis=self.axis) + np.sum(border_pred, axis=self.axis)
        return numerator / denominator

    def measured_average_distance(self):
        """
        This function returns the average symmetric surface distance (ASSD) between prediction and reference

        :return: assd
        """
        if self.smooth_dr == 0 and np.sum(self.pred + self.ref) == 0:
            return 0
        ref_border_dist, pred_border_dist, ref_border, pred_border = \
            self.border_distance()
        assd = \
            (np.sum(ref_border_dist, axis=self.axis) + np.sum(pred_border_dist, axis=self.axis)) \
            / (np.sum(pred_border + ref_border, axis=self.axis) + self.smooth_dr)
        return assd

    def measured_masd(self):
        """
        This function returns the mean average surfance distance (MASD) between prediction and reference

        :return: masd
        """
        if self.smooth_dr == 0 and (np.sum(self.pred) == 0 or np.sum(self.ref) == 0):
            return 0
        ref_border_dist, pred_border_dist, ref_border, pred_border = \
            self.border_distance()
        masd = 0.5 * (
            np.sum(ref_border_dist, axis=self.axis) / (np.sum(pred_border, axis=self.axis) + self.smooth_dr)
            + np.sum(pred_border_dist, axis=self.axis) / (np.sum(ref_border, axis=self.axis) + self.smooth_dr)
        )
        return masd

    def measured_hausdorff_distance(self):
        """
        This function returns the Hausdorff distance between prediction and reference

        :return: hd
        """
        ref_border_dist, pred_border_dist, _, _ = \
            self.border_distance()
        hd = np.max(np.maximum(ref_border_dist, pred_border_dist), axis=self.axis)
        return hd

    def measured_hausdorff_distance_perc(self):
        """
        This function returns the xth percentile Hausdorff distance between prediction and reference

        :return: hdp
        """
        if "hd_perc" in self.dict_args.keys():
            perc = self.dict_args["hd_perc"]
        else:
            perc = 95
        ref_border_dist, pred_border_dist, _, _ = \
            self.border_distance()
        msk = self.ref + self.pred > 0
        ref_border_dist[~msk] = np.nan
        pred_border_dist[~msk] = np.nan
        hdp = np.maximum(
            np.nanpercentile(ref_border_dist, q=perc, axis=self.axis),
            np.nanpercentile(pred_border_dist, q=perc, axis=self.axis),
        )
        return hdp

    def to_dict_meas(self, fmt="{:.4f}"):
        result_dict = {}
        for key in self.measures:
            if len(self.measures_dict[key]) == 2:
                result = self.measures_dict[key][0]()
            else:
                result = self.measures_dict[key][0](self.measures_dict[key][2])
            result_dict[key] = fmt.format(result)
        return result_dict  # trim the last comma

    def list_labels(self):
        if self.list_labels is None:
            return ()
        return tuple(np.unique(self.list_labels))

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
