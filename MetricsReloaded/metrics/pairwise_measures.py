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

    def __init__(self, pred, ref, list_values, measures=[], dict_args={}):
        self.pred = np.asarray(pred, dtype=np.int32)
        self.ref = np.asarray(ref, dtype=np.int32)
        self.dict_args = dict_args
        self.list_values = list_values
        self.measures = measures
        self.measures_dict = {
            "mcc": (self.matthews_correlation_coefficient, "MCC"),
            "wck": (self.weighted_cohens_kappa, "WCK"),
            "ba": (self.balanced_accuracy, "BAcc"),
            "ec": (self.expected_cost, "EC"),
        }

    def expected_cost(self):
        cm = self.confusion_matrix()
        priors = np.sum(cm, 0) / np.sum(cm)
        numb_perc = np.sum(cm, 0)
        rmatrix = cm / numb_perc
        prior_matrix = np.tile(priors, [cm.shape[0], 1])
        priorbased_weights = 1.0 / (cm.shape[1] * prior_matrix)
        for c in range(cm.shape[0]):
            priorbased_weights[c, c] = 0
        if "ec_costs" in self.dict_args.keys():
            weights = self.dict_args["ec_costs"]
        else:
            weights = priorbased_weights
        ec = np.sum(prior_matrix * weights * rmatrix)
        return ec

    def best_naive_ec(self):
        cm = self.confusion_matrix()
        priors = np.sum(cm, 0) / np.sum(cm)
        prior_matrix = np.tile(priors, [cm.shape[0], 1])
        priorbased_weights = 1 / (cm.shape[1] * prior_matrix)
        for c in range(cm.shape[0]):
            priorbased_weights[c, c] = 0

        if "ec_costs" in self.dict_args.keys():
            weights = self.dict_args["ec_costs"]
        else:
            weights = priorbased_weights
        total_cost = np.sum(weights * prior_matrix, 1)
        return np.min(total_cost)

    def normalised_expected_cost(self):
        naive_cost = self.best_naive_ec()
        ec = self.expected_cost()
        return ec / naive_cost

    def matthews_correlation_coefficient(self):
        """
        Calculates the multiclass Matthews Correlation Coefficient defined as

        Brian W Matthews. 1975. Comparison of the predicted and observed secondary structure of T4 phage lysozyme.
        Biochimica et Biophysica Acta (BBA)-Protein Structure 405, 2 (1975), 442–451.

        .. math::

            R_k = \dfrac{cov_k(Pred,Ref)}{\sqrt{cov_k(Pred,Pred)*cov_k(Ref,Ref)}}

        with

        .. math::
            cov_k(X,Y) = \dfrac{1}{K}\sum_{k=1}^{K}cov(X_k,Y_k)

        :return: Matthews Correlation Coefficient
        """
        one_hot_pred = one_hot_encode(self.pred, len(self.list_values))
        one_hot_ref = one_hot_encode(self.ref, len(self.list_values))
        cov_pred = 0
        cov_ref = 0
        cov_pr = 0
        for f in range(len(self.list_values)):
            cov_pred += np.cov(one_hot_pred[:, f], one_hot_pred[:, f])[0, 1]
            cov_ref += np.cov(one_hot_ref[:, f], one_hot_ref[:, f])[0, 1]
            cov_pr += np.cov(one_hot_pred[:, f], one_hot_ref[:, f])[0, 1]
        
        numerator = cov_pr
        denominator = np.sqrt(cov_pred * cov_ref)
        return numerator / denominator

    def chance_agreement_probability(self):
        """Determines the probability of agreeing by chance given two classifications.
        To be used for CK calculation

        return: chance (probability for classification of agreeing by chance)
        """
        chance = 0
        for f in self.list_values:
            prob_pred = len(np.where(self.pred == f)[0]) / np.size(self.pred)
            prob_ref = len(np.where(self.ref == f)[0]) / np.size(self.ref)
            chance += prob_pred * prob_ref
        return chance

    def confusion_matrix(self):
        """
        Provides the confusion matrix Prediction in rows, Reference in columns

        :return: confusion_matrix
        """
        one_hot_pred = one_hot_encode(self.pred, len(self.list_values))
        one_hot_ref = one_hot_encode(self.ref, len(self.list_values))
        confusion_matrix = np.matmul(one_hot_pred.T, one_hot_ref)
        return confusion_matrix

    def balanced_accuracy(self):
        """Calculation of balanced accuracy as average of correctly classified
        by reference class across all classes

        .. math::

            BA = \dfrac{\sum_{k=1}^{K} \dfrac{TP_k}{TP_k+FN_k}}{K}

        :return: balanced_accuracy
        """
        cm = self.confusion_matrix()
        col_sum = np.sum(cm, 0)
        numerator = np.sum(np.diag(cm) / col_sum)
        denominator = len(self.list_values)
        balanced_accuracy = numerator / denominator
        return balanced_accuracy

    def expectation_matrix(self):
        """
        Determination of the expectation matrix to be used for CK derivation

        :return: expectation_matrix

        """
        one_hot_pred = one_hot_encode(self.pred, len(self.list_values))
        one_hot_ref = one_hot_encode(self.ref, len(self.list_values))
        pred_numb = np.sum(one_hot_pred, 0)
        ref_numb = np.sum(one_hot_ref, 0)
        expectation_matrix = np.matmul(np.reshape(pred_numb, [-1, 1]), np.reshape(ref_numb, [1, -1]))/ np.shape(one_hot_pred)[0]
        return expectation_matrix

    def weighted_cohens_kappa(self):
        """

        Derivation of weighted cohen's kappa. The weight matrix is set to 1-ID(len(list_values))
        - cost of 1 for each error type if no weight provided

        :return: weighted_cohens_kappa

        """
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
        weighted_cohens_kappa = 1 - numerator / denominator
        return weighted_cohens_kappa

    def to_dict_meas(self, fmt="{:.4f}"):
        """Given the selected metrics provides a dictionary with relevant metrics"""
        result_dict = {}
        for key in self.measures:
            result = self.measures_dict[key][0]()
            result_dict[key] = result
        return result_dict  


class BinaryPairwiseMeasures(object):
    def __init__(
        self,
        pred,
        ref,
        measures=[],
        connectivity_type=1,
        pixdim=None,
        empty=False,
        dict_args={},
    ):

        self.measures_dict = {
            "numb_ref": (self.n_pos_ref, "NumbRef"),
            "numb_pred": (self.n_pos_pred, "NumbPred"),
            "numb_tp": (self.n_intersection, "NumbTP"),
            "numb_fp": (self.fp, "NumbFP"),
            "numb_fn": (self.fn, "NumbFN"),
            "accuracy": (self.accuracy, "Accuracy"),
            "nb": (self.net_benefit_treated, "NB"),
            "ec": (self.normalised_expected_cost, "ECn"),
            "ba": (self.balanced_accuracy, "BalAcc"),
            "cohens_kappa": (self.cohens_kappa, "CohensKappa"),
            "lr+": (self.positive_likelihood_ratio, "LR+"),
            "iou": (self.intersection_over_union, "IoU"),
            "fbeta": (self.fbeta, "FBeta"),
            "dsc":(self.dsc, "DSC"),
            "youden_ind": (self.youden_index, "YoudenInd"),
            "ppv":(self.positive_predictive_value,'PPV'),
            "npv":(self.negative_predictive_value,'NPV'),
            "ior":(self.intersection_over_reference,"IoR"),
            "sensitivity":(self.sensitivity,"Sens"),
            "specificity":(self.specificity,"Spec"),
            "mcc": (self.matthews_correlation_coefficient, "MCC"),
            "cldice": (self.centreline_dsc, "CentreLineDSC"),
            "assd": (self.measured_average_distance, "ASSD"),
            "boundary_iou": (self.boundary_iou, "BoundaryIoU"),
            "hd": (self.measured_hausdorff_distance, "HD"),
            "hd_perc": (self.measured_hausdorff_distance_perc, "HDPerc"),
            "masd": (self.measured_masd, "MASD"),
            "nsd": (self.normalised_surface_distance, "NSD"),
            "avdr": (self.absolute_volume_difference_ratio, "AVDR")
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
        self.connectivity = connectivity_type
        self.pixdim = pixdim
        self.dict_args = dict_args

    def __fp_map(self):
        """
        This function calculates the false positive map

        :return: FP map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        fp_map = np.asarray((pred_float - ref_float) > 0.0, dtype=np.float32)
        return fp_map
    
    def __fn_map(self):
        """
        This function calculates the false negative map

        :return: FN map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        fn_map = np.asarray((ref_float - pred_float) > 0.0, dtype=np.float32)
        return fn_map
    
    def __tp_map(self):
        """
        This function calculates the true positive map

        :return: TP map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        tp_map = np.asarray((ref_float + pred_float) > 1.0, dtype=np.float32)
        return tp_map

    def __tn_map(self):
        """
        This function calculates the true negative map

        :return: TN map
        """
        ref_float = np.asarray(self.ref, dtype=np.float32)
        pred_float = np.asarray(self.pred, dtype=np.float32)
        tn_map = np.asarray((ref_float + pred_float) < 0.5, dtype=np.float32)
        return tn_map
    
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
        intersection_map = np.multiply(self.ref, self.pred)
        return intersection_map

    @CacheFunctionOutput
    def n_pos_ref(self):
        """
        Returns the number of elements in ref
        
        :return: n_pos_ref
        """
        n_pos_ref = np.sum(self.ref)
        return n_pos_ref

    @CacheFunctionOutput
    def n_neg_ref(self):
        """
        Returns the number of negative elements in ref
        
        :return: n_neg_ref
        """
        n_neg_ref = np.sum(1-self.ref)
        return n_neg_ref

    @CacheFunctionOutput
    def n_pos_pred(self):
        """
        Returns the number of positive elements in the prediction
        
        :return: n_pos_pred
        """
        n_pos_pred = np.sum(self.pred)
        return np.sum(self.pred)

    @CacheFunctionOutput
    def n_neg_pred(self):
        """
        Returns the number of negative elements in the prediction
        
        :return: n_neg_pred
        """
        n_neg_pred = np.sum(1-self.pred)
        return n_neg_pred

    @CacheFunctionOutput
    def fp(self):
        """
        Calculates the number of FP as sum of elements in FP_map
        
        :return: fp
        """
        fp = np.sum(self.__fp_map())
        return fp

    @CacheFunctionOutput
    def fn(self):
        """
        Calculates the number of FN as sum of elements of FN_map
        
        :return: fn
        """
        fn = np.sum(self.__fn_map())
        return fn

    @CacheFunctionOutput
    def tp(self):
        """
        Returns the number of true positive (TP) elements
        
        :return: tp
        """
        tp = np.sum(self.__tp_map())
        return tp

    @CacheFunctionOutput
    def tn(self):
        """
        Returns the number of True Negative (TN) elements
        
        :return: tn
        """
        tn = np.sum(self.__tn_map())
        return tn

    @CacheFunctionOutput
    def n_intersection(self):
        """
        Returns the number of elements in the intersection of reference and prediction (=TP)
        
        .. math::

            I =  TP


        :return: n_intersection
        """
        n_intersection = np.sum(self.__intersection_map())
        return n_intersection

    @CacheFunctionOutput
    def n_union(self):
        """
        Returns the number of elements in the union of reference and prediction

        .. math::

            U = {\vert} Pred {\vert} + {\vert} Ref {\vert} - TP

        :return: n_union
        """
        n_union = np.sum(self.__union_map())
        return n_union

    def youden_index(self):
        """
        Calculates the Youden Index (YI) defined as:

        .. math::

            YI = Specificity + Sensitivity - 1

        Youden, W.J, Index for rating diagnostic tests - 1950 Cancer 3 - 32,35

        :return: youden_index
        """
        youden_index =  self.specificity() + self.sensitivity() - 1
        return youden_index

    def sensitivity(self):
        """
        Calculates the sensitivity defined as

        .. math::

            Sens = \dfrac{TP}{|Ref|}

        Yerushalmy J., Statistical Problems in assessing Methods of Medical Diagnosis with Special reference to X-Ray Techniques, 1947, Public Health Reports, pp1432-1449

        This measure is not defined for empty reference. Will raise a warning and return a nan value

        :return: sensitivity
        """
        if self.n_pos_ref() == 0:
            warnings.warn("reference empty, sensitivity not defined")
            return np.nan
        sensitivity = self.tp() / self.n_pos_ref()
        return sensitivity

    def specificity(self):
        """
        Calculates the specificity defined as

        .. math::

            Spec = \dfrac{TN}{|1-Ref|}

        Yerushalmy J., Statistical Problems in assessing Methods of Medical Diagnosis with Special reference to X-Ray Techniques, 1947, Public Health Reports, pp1432-1449

        This measure is not defined when there is no reference negative. This will
        raise a warning and return a nan

        :return: specificity
        """
        if self.n_neg_ref() == 0:
            warnings.warn("reference all positive, specificity not defined")
            return np.nan
        specificity = self.tn() / self.n_neg_ref()
        return specificity

    def balanced_accuracy(self):
        """
        Calculates and returns the balanced accuracy defined for the
        binary case as the average between sensitivity and specificity

        Margherita Grandini, Enrico Bagli, and Giorgio Visani. 2020. Metrics for multi-class classification: an overview. arXiv
        preprint arXiv:2008.05756 (2020).

        :return: balanced accuracy
        """
        balanced_accuracy = 0.5 * self.sensitivity() + 0.5 * self.specificity()
        return balanced_accuracy

    def accuracy(self):
        """
        Calculate and returns the accuracy defined as

        Margherita Grandini, Enrico Bagli, and Giorgio Visani. 2020. Metrics for multi-class classification: an overview. arXiv
        preprint arXiv:2008.05756 (2020).

        .. math::

            Acc = \dfrac{TN+TP}{TN+TP+FN+FP}

        :return: accuracy
        """
        accuracy = (self.tn() + self.tp()) / (self.tn() + self.tp() + self.fn() + self.fp())
        return accuracy

    def false_positive_rate(self):
        """
        Calculates and returns the false positive rate defined as

        .. math::

            FPR = \dfrac{FP}{|1-Ref|}

        Burke D, Brundage J, Redfield R., Measurement of the False positive rate in a screening Program for Human Immunodeficiency Virus Infections - 1988 - The New England Journal of Medicine 319 (15) 961-964

        :return: false_positive_rate
        """
        false_positive_rate = self.fp() / self.n_neg_ref()
        return false_positive_rate

    def normalised_expected_cost(self):
        """
        Calculates and returns the normalised expected cost

        Luciana Ferrer. 2022. Analysis and Comparison of Classification Metrics. arXiv preprint arXiv:2209.05355 (2022).

        :return: normalised_expected_cost
        """

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
        r_fp = self.fp() / self.n_neg_ref()
        r_fn = self.fn() / self.n_pos_ref()
        if alpha >= 1:
            normalised_expected_cost = alpha * r_fp + r_fn
        else:
            normalised_expected_cost = r_fp + 1 / alpha * r_fn
        return normalised_expected_cost

    def matthews_correlation_coefficient(self):
        """
        Calculates and returns the MCC for the binary case

        Matthews, B.W. Comparison of the predicted and observed secondary structure of T4 phage lysozyme. Biochimica et Biophysica Acta - Protein Structure - 1975 405 442-451

        .. math::

            MCC = \dfrac{TP * TN - FP * FN}{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}

        :return: mcc
        """
        numerator = self.tp() * self.tn() - self.fp() * self.fn()
        denominator = (
            (self.tp() + self.fp())
            * (self.tp() + self.fn())
            * (self.tn() + self.fp())
            * (self.tn() + self.fn())
        )
        mcc = numerator / np.sqrt(denominator)
        return mcc

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
        """
        Calculates and return the Cohen's kappa score defined as

        .. math::

            CK = \dfrac{p_o - p_e}{1-p_e}

        where :math: `p_e = ` expected chance matching and :math: `p_o = `observed accuracy

        Cohen, J. A coefficient of agreement for nominal scales - Educational and Psychological Measurement (1960) 20 37-46

        :return: cohens_kappa

        """
        p_e = self.expected_matching_ck()
        p_o = self.accuracy()
        numerator = p_o - p_e
        denominator = 1 - p_e
        cohens_kappa = numerator / denominator
        return cohens_kappa

    def positive_likelihood_ratio(self):
        """
        Calculates the positive likelihood ratio

        John Attia. 2003. Moving beyond sensitivity and specificity: using likelihood ratios to help interpret diagnostic tests.
        Australian prescriber 26, 5 (2003), 111–113.

        .. math::

            LR+ = \dfrac{Sensitivity}{1-Specificity}

        :return: positive_likelihood_ratio (LR+)
        """
        numerator = self.sensitivity()
        denominator = 1 - self.specificity()
        if self.n_neg_ref() == 0:
            warnings.warn("reference all positive, specificity not defined")
            return np.nan
        if self.n_pos_ref() == 0:
            warnings.warn("reference empty - sensitivity not defined")
            return np.nan
        if self.specificity() == 1:
            warnings.warn("Perfect specifiicty - likelihood ratio not defined")
            return np.nan
        positive_likelihood_ratio = numerator / denominator
        return positive_likelihood_ratio

    def pred_in_ref(self):
        """
        Determines if prediction and reference overlap on at least one voxel.

        :return: 1 if true, 0 otherwise
        """
        intersection = np.sum(self.pred * self.ref)
        if intersection > 0:
            return 1
        else:
            return 0

    def positive_predictive_value(self):
        """
        Calculates the positive predictive value

        .. math::

            PPV = \dfrac{TP}{TP+FP}

        Not defined when no positives in the prediction - returns nan if both
        reference and prediction empty. Returns 0 if only prediction empty

        Fletcher, R.H and Fletcher S.W (2005) - Clinical Epidemiology, the essentials p45

        :return: positive_predictive_value (PPV)
        """
        if self.flag_empty_pred:
            if self.flag_empty_ref:
                warnings.warn("ref and prediction empty ppv not defined")
                return np.nan
            else:
                warnings.warn("prediction empty, ppv not defined but set to 0")
                return 0
        positive_predictive_value = self.tp() / (self.tp() + self.fp())
        return positive_predictive_value

    def recall(self):
        """
        Calculates and returns recall = sensitivity

        :return: Recall = Sensitivity
        """
        if self.n_pos_ref() == 0:
            warnings.warn("reference is empty, recall not defined")
            return np.nan
        if self.n_pos_pred() == 0:
            warnings.warn(
                "prediction is empty but ref not, recall not defined but set to 0"
            )
            return 0
        recall = self.tp() / (self.tp() + self.fn())
        return recall

    def dsc(self):
        """
        Calculates the Dice Similarity Coefficient defined as

        Lee R Dice. 1945. Measures of the amount of ecologic association between species. Ecology 26, 3 (1945), 297–302.

        ..math::

            DSC = \dfrac{2TP}{2TP+FP+FN}

        
        This is also F:math:`{\\beta}` for :math:`{\\beta}`=1

        :return: dsc

        """

        numerator = 2 * self.tp()
        denominator = self.n_pos_pred() + self.n_pos_ref()
        if denominator == 0:
            warnings.warn("Both Prediction and Reference are empty - set to 1 as correct solution even if not defined")
            return 1
        else:
            dsc = numerator / denominator
            return dsc

    def fbeta(self):
        """
        Calculates FBeta score defined as

        Nancy Chinchor. 1992. MUC-4 Evaluation Metrics. In Proceedings of the 4th Conference on Message Understanding
        (McLean, Virginia) (MUC4 ’92). Association for Computational Linguistics, USA, 22–29. https://doi.org/10.3115/
        1072064.1072067

        .. math::

            F_{\\beta} = (1+{\\beta}^2) \dfrac{Precision  * Recall}{{\\beta}^2 * Precision + recall}

        When :math:`{\\beta}=1` it corresponds to the dice score. The :math:`{\\beta}` parameter is
        set up in the class dictionary of options

        :return: fbeta value

        """
        if "beta" in self.dict_args.keys():
            beta = self.dict_args["beta"]
        else:
            warnings.warn("beta value not specified in option - default set to 1")
            beta = 1
        numerator = (
            (1 + np.square(beta)) * self.positive_predictive_value() * self.recall()
        )
        denominator = (
            np.square(beta) * self.positive_predictive_value() + self.recall()
        )
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
            fbeta = numerator / denominator
            return fbeta

    def net_benefit_treated(self):
        """
        This functions calculates the net benefit treated according to a specified exchange rate

        Andrew J Vickers, Ben Van Calster, and Ewout W Steyerberg. 2016. Net benefit approaches to the evaluation of
        prediction models, molecular markers, and diagnostic tests. bmj 352 (2016).

        .. math::

            NB = \dfrac{TP}{TP+TN+FP+FN} - \dfrac{FP}{TP+TN+FP+FN} * ER

        where ER relates to the exchange rate. For instance if a suitable exchange rate is to find
        1 positive case among 10 tested (1TP for 9 FP), the exchange rate would be 1/9

        :return: net_benefit
        """
        if "exchange_rate" in self.dict_args.keys():
            er = self.dict_args["exchange_rate"]
        else:
            er = 1
        n = np.size(self.pred)
        tp = self.tp()
        fp = self.fp()
        net_benefit = tp / n - fp / n * er
        return net_benefit

    def negative_predictive_value(self):
        """
        This function calculates the negative predictive value ratio between
        the number of true negatives and the total number of negative elements

        Fletcher, R.H and Fletcher S.W (2005) - Clinical Epidemiology, the essentials p45

        .. math::

            NPV = \dfrac{TN}{N}

        :return: NPV
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
        negative_predictive_value = self.tn() / (self.fn() + self.tn())
        return negative_predictive_value


    def fppi(self):
        """
        This function returns the average number of false positives per
         image, assuming that the cases are collated on the last axis of the array

        Bram Van Ginneken, Samuel G Armato III, Bartjan de Hoop, Saskia van Amelsvoort-van de Vorst, Thomas Duindam, Meindert Niemeijer, Keelin Murphy, Arnold Schilham, Alessandra Retico, Maria Evelina Fantacci, et al. Comparing and combining algorithms for computer-aided detection of pulmonary nodules in computed tomography scans: the anode09 study. Medical image analysis, 14(6):707–722, 2010.
        Andriy I Bandos, Howard E Rockette, Tao Song, and David Gur. Area under the free-response roc curve (froc) and a related summary index. Biometrics, 65(1):247–256, 2009.
        
        """
        sum_per_image = np.sum(
            np.reshape(self.__fp_map(), -1, self.ref.shape[-1]), axis=0
        )
        fppi = np.mean(sum_per_image)
        return fppi

    def intersection_over_reference(self):
        """
        This function calculates the ratio of the intersection of prediction and
        reference over reference.

        Pavel Matula, Martin Maška, Dmitry V Sorokin, Petr Matula, Carlos Ortiz-de Solórzano, and Michal Kozubek. Cell tracking accuracy measurement based on comparison of acyclic oriented graphs. PloS one, 10(12):e0144959, 2015.

        .. math::

            IoR = \dfrac{| \text{Pred} \cap \text{Ref} |}{| Ref |}

        :return: IoR

        """
        if self.flag_empty_ref:
            warnings.warn("IoR not defined - Empty reference")
            return np.nan
        ior = self.n_intersection()/self.n_pos_ref()
        return ior

    def intersection_over_union(self):
        """
        This function calculates the intersection of prediction and
        reference over union - This is also the definition of
        jaccard coefficient

        Murphy, A.H. The Finley Affair: a signal event in the history of forecast verification - Weather and Forecasting (1996) 11

        .. math::

            IoU = \dfrac{|Pred \cap Ref|}{| Pred \cup Ref |}

        :return: IoU

        """
        if self.flag_empty_pred and self.flag_empty_ref:
            warnings.warn("IoU not defined - Both reference and prediction are empty")
            return np.nan
        iou = self.n_intersection() / self.n_union()
        return iou

    def com_dist(self):
        """
        This function calculates the euclidean distance between the centres
        of mass of the reference and prediction.

        
        :return: Euclidean distance between centre of mass when reference and prediction not empty
        -1 otherwise

        """
        
        if self.flag_empty_pred or self.flag_empty_ref:
            warnings.warn('Impossible to calculate distance between centre of masses as either reference of prediction is empty')
            return -1
        else:
            com_ref = compute_center_of_mass(self.ref)
            com_pred = compute_center_of_mass(self.pred)

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

        :return: com_ref - Centre of mass coordinates of reference when not empty, -1 otherwise
        """
        if self.flag_empty_ref:
            warnings.warn('Empty reference - centre of mass not defined')
            return -1
        com_ref = ndimage.center_of_mass(self.ref)
        return com_ref

    def com_pred(self):
        """
        This functions provides the centre of mass of the predmented element
        :returns: -1 if empty image, centre of mass of prediction otherwise
        """
        if self.flag_empty_pred:
            warnings.warn('Empty prediction - centre of mass not defined')
            return -1
        else:
            com_pred = ndimage.center_of_mass(self.pred)
            return com_pred

    def list_labels(self):
        """
        Creates the tuple with unique values of labels

        return list_labels
        """
        if self.list_labels is None:
            return ()
        return tuple(np.unique(self.list_labels))

    def absolute_volume_difference_ratio(self):
        """
        This function calculates the ratio of difference in volume between
        the reference and prediction images.

        .. math::

            AVDR = \dfrac{| Pred - Ref|}{| Ref |} 

        :return: avdr

        """
        if self.n_pos_ref() == 0:
            warnings.warn('Empty reference - absolute volume difference ratio not defined')
            return np.nan
        avdr = np.abs(self.n_pos_ref() - self.n_pos_pred()) / self.n_pos_ref()
        return avdr

    @CacheFunctionOutput
    def skeleton_versions(self):
        """
        Creates the skeletonised version of both reference and prediction

        :return: skeleton_ref, skeleton_pred
        """
        skeleton_ref = compute_skeleton(self.ref)
        skeleton_pred = compute_skeleton(self.pred)
        return skeleton_ref, skeleton_pred

    def topology_precision(self):
        """
        Calculates topology precision defined as

        .. math::

            Prec_{Top} = \dfrac{|S_{Pred} \cap Ref|}{|S_{Pred}|}

        with :math:`S_{Pred}` the skeleton of Pred

        :return: topology_precision
        """
        skeleton_ref, skeleton_pred = self.skeleton_versions()
        numerator = np.sum(skeleton_pred * self.ref)
        denominator = np.sum(skeleton_pred)
        if denominator == 0:
            warnings.warn('Empty prediction skeleton - topology precision not defined')
            return np.nan
        topology_precision = numerator / denominator
        return topology_precision

    def topology_sensitivity(self):
        """
        Calculates the topology sensitivity defined as

        .. math::

            Sens_{Top} = \dfrac{|S_{Ref} \cap Pred|}{|S_{Ref}|}

        with :math:`S_{Ref}` the skeleton of Ref

        :return: topology_sensitivity
        """
        skeleton_ref, skeleton_pred = self.skeleton_versions()
        numerator = np.sum(skeleton_ref * self.pred)
        denominator = np.sum(skeleton_ref)
        if denominator == 0:
            warnings.warn("Reference skeleton empty - topology sensitivity not defined")
            return np.nan
        topology_sensitivity = numerator / denominator
        return topology_sensitivity

    def centreline_dsc(self):
        """
        Calculates the centre line dice score defined as

        Suprosanna Shit, Johannes C Paetzold, Anjany Sekuboyina, Ivan Ezhov, Alexander Unger, Andrey Zhylka, Josien PW
        Pluim, Ulrich Bauer, and Bjoern H Menze. 2021. clDice-a novel topology-preserving loss function for tubular structure
        segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16560–16569

        .. math::

            cDSC = 2\dfrac{Sens_{Top} * Prec_{Top}}{Sens_{Top} + Prec_{Top}}

        :return: cDSC
        """
        if self.n_pos_pred == 0 and self.n_pos_ref == 0:
            warnings.warn("Both reference and prediction are empty - setting to max")
            return 1
        top_prec = self.topology_precision()
        top_sens = self.topology_sensitivity()
        numerator = 2 * top_sens * top_prec
        denominator = top_sens + top_prec
        if np.isnan(top_sens) or np.isnan(top_sens):
            warnings.warn("Topology sensitivity or precision not defined")
            return np.nan
        cDSC = numerator / denominator
        return cDSC

    def boundary_iou(self):
        """
        This functions determines the boundary iou

        Bowen Cheng, Ross Girshick, Piotr Dollár, Alexander C Berg, and Alexander Kirillov. 2021. Boundary IoU: Improving
Object-Centric Image Segmentation Evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 15334–15342.

        .. math::

            B_{IoU}(A,B) = \dfrac{| A_{d} \cap B_{d} |}{|A_d| + |B_d| - |A_d \cap B_d|}

        where :math:A_d are the pixels of A within a distance d of the boundary
        :return: boundary_iou

        """
        if "boundary_dist" in self.dict_args.keys():
            distance = self.dict_args["boundary_dist"]
        else:
            distance = 1
        if int(self.n_pos_ref()) == 0 and int(self.n_pos_pred()) == 0:
            warnings.warn("Both prediction and reference empty - setting to max for boudnary ioU")
            return 1
        else:
            border_ref = MorphologyOps(self.ref, self.connectivity).border_map()
            distance_border_ref = ndimage.distance_transform_edt(1 - border_ref)

            border_pred = MorphologyOps(self.pred, self.connectivity).border_map()
            distance_border_pred = ndimage.distance_transform_edt(1 - border_pred)

            lim_dbp = np.where(
                np.logical_and(distance_border_pred < distance, self.pred>0),
                np.ones_like(border_pred),
                np.zeros_like(border_pred),
            )
            lim_dbr = np.where(
                np.logical_and(distance_border_ref < distance, self.ref>0),
                np.ones_like(border_ref),
                np.zeros_like(border_ref),
            )

            intersect = np.sum(lim_dbp * lim_dbr)
            union = np.sum(
                np.where(
                    lim_dbp + lim_dbr > 0,
                    np.ones_like(border_ref),
                    np.zeros_like(border_pred),
                )
            )
            # if union == 0:
            #     warnings.warn('Union empty for boundary iou - not defined')
            #     return np.nan
            boundary_iou = intersect / union
            return boundary_iou


    @CacheFunctionOutput
    def border_distance(self):
        """
        This functions determines the map of distance from the borders of the
        prediction and the reference and the border maps themselves

        :return: distance_border_ref, distance_border_pred, border_ref,
        border_pred
        """
        border_ref = MorphologyOps(self.ref, self.connectivity).border_map()
        border_pred = MorphologyOps(self.pred, self.connectivity).border_map()
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
        """
        Calculates the normalised surface distance (NSD) between prediction and reference
        using the distance parameter :math:`{\\tau}`

        Stanislav Nikolov, Sam Blackwell, Alexei Zverovitch, Ruheena Mendes, Michelle Livne, Jeffrey De Fauw, Yojan Patel,
        Clemens Meyer, Harry Askham, Bernadino Romera-Paredes, et al. 2021. Clinically applicable segmentation of head
        and neck anatomy for radiotherapy: deep learning algorithm development and validation study. Journal of Medical
        Internet Research 23, 7 (2021), e26151.

        .. math::

            NSD(A,B)^{(\\tau)} = \dfrac{|S_{A} \cap Bord_{B,\\tau}| + |S_{B} \cup Bord_{A,\\tau}|}{|S_{A}| + S_{B}}

        :return: NSD
        """
        if "nsd" in self.dict_args.keys():
            tau = self.dict_args["nsd"]
        else:
            warnings.warn('No value set up for NSD tolerance - default to 1')
            tau = 1
        if int(self.n_pos_pred()) == 0 and int(self.n_pos_ref()) == 0 :
            warnings.warn("Both reference and prediction are empty - setting to best")
            return 1
        else:
            dist_ref, dist_pred, border_ref, border_pred = self.border_distance()
            reg_ref = np.where(
            dist_ref <= tau, np.ones_like(dist_ref), np.zeros_like(dist_ref)
            )
            reg_pred = np.where(
                dist_pred <= tau, np.ones_like(dist_pred), np.zeros_like(dist_pred)
            )
            # print(np.sum(border_pred),np.sum(reg_ref),np.sum(border_ref),np.sum(reg_pred))
            # print(np.sum(border_pred*reg_ref),np.sum(border_ref*reg_pred))
            numerator = np.sum(border_pred * reg_ref) + np.sum(border_ref * reg_pred)
            denominator = np.sum(border_ref) + np.sum(border_pred)
            # print(numerator, denominator, tau)
            return numerator / denominator

    def measured_distance(self):
        """
        This functions calculates the average symmetric distance and the
        hausdorff distance between a prediction and a reference image

        :return: hausdorff distance and average symmetric distance, hausdorff distance at perc
        and masd

        """
     
        if "hd_perc" in self.dict_args.keys():
            perc = self.dict_args["hd_perc"]
        else:
            warnings.warn('Percentile not specified in options for Hausdorff distance - default set to 95')
            perc = 95
        if np.sum(self.pred + self.ref) == 0:
            warnings.warn("Prediction and reference empty - distances set to 0")
            return 0, 0, 0, 0
        if np.sum(self.pred) == 0 and np.sum(self.ref)>0:
            warnings.warn("Prediction empty but reference not empty - need to set to worse case in aggregation")
            return np.nan, np.nan, np.nan, np.nan
        if np.sum(self.ref) == 0 and np.sum(self.pred)>0:
            warnings.warn('Prediction not empty but reference empty - non existing output - need be set to WORSE case in aggregation')
            return np.nan, np.nan, np.nan, np.nan
        (
            ref_border_dist,
            pred_border_dist,
            ref_border,
            pred_border,
        ) = self.border_distance()
        # print(ref_border_dist)
        average_distance = (np.sum(ref_border_dist) + np.sum(pred_border_dist)) / (
            np.sum(pred_border + ref_border)
        )
        masd = 0.5 * (
            np.sum(ref_border_dist) / np.sum(pred_border)
            + np.sum(pred_border_dist) / np.sum(ref_border)
        )

        hausdorff_distance = np.max([np.max(ref_border_dist), np.max(pred_border_dist)])
       
        hausdorff_distance_perc = np.max(
            [
                np.percentile(ref_border_dist[pred_border > 0], q=perc),
                np.percentile(pred_border_dist[ref_border > 0], q=perc),
            ]
        )


        return hausdorff_distance, average_distance, hausdorff_distance_perc, masd

    def measured_average_distance(self):
        """
        This function returns only the average symmetric surface distance when calculating the
        distances between prediction and reference

        .. math::

            ASSD(A,B) = \dfrac{\sum_{a\in A}d(a,B) + \sum_{b\in B}d(b,A)}{|A|+ |B|}

        Heimann, T., et al. (2009), Comparison and evaluation of methods for liver segmentation from CT datasets. IEEE Trans Med Imaging. 28(8): p. 1251-65.
        Varduhi Yeghiazaryan and Irina Voiculescu. An overview of current evaluation methods used in medical image segmentation. Department of Computer Science, University of Oxford, 2015.

        :return: assd
        """
        assd = self.measured_distance()[1]
        return assd

    def measured_masd(self):
        """
        This function returns only the mean average surface distance defined as
        
        Miroslav Beneš and Barbara Zitová. 2015. Performance evaluation of image segmentation algorithms on microscopic
        image data. Journal of microscopy 257, 1 (2015), 65–85.

        .. math::

            MASD(A,B) = \dfrac{1}{2}(\dfrac{\sum_{a\in A}d(a,B)}{|A|} + \dfrac{\sum_{b\in B}d(b,A)}{|B|})
        
            
        :return: masd

        """
        masd = self.measured_distance()[3]
        return masd

    def measured_hausdorff_distance(self):
        """
        This function returns only the hausdorff distance when calculated the
        distances between prediction and reference

        Daniel P Huttenlocher, Gregory A. Klanderman, and William J Rucklidge. 1993. Comparing images using the Hausdorff
        distance. IEEE Transactions on pattern analysis and machine intelligence 15, 9 (1993), 850–863.

        :return: hausdorff_distance
        """
        hausdorff_distance = self.measured_distance()[0]
        return hausdorff_distance

    def measured_hausdorff_distance_perc(self):
        """
        This function returns the xth percentile hausdorff distance

        Daniel P Huttenlocher, Gregory A. Klanderman, and William J Rucklidge. 1993. Comparing images using the Hausdorff
        distance. IEEE Transactions on pattern analysis and machine intelligence 15, 9 (1993), 850–863.
        
        :return: hausdorff_distance_perc
        """
        hausdorff_distance_perc = self.measured_distance()[2]
        return hausdorff_distance_perc

    def to_dict_meas(self, fmt="{:.4f}"):
        result_dict = {}
        for key in self.measures:
            if len(self.measures_dict[key]) == 2:
                result = self.measures_dict[key][0]()
            else:
                result = self.measures_dict[key][0](self.measures_dict[key][2])
            result_dict[key] = result
        return result_dict  

    