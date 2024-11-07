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
Calibration measures - :mod:`MetricsReloaded.metrics.calibration_measures`
==========================================================================

This module provides classes for calculating :ref:`calibration
<calibration>` measures.

.. _calibration:

Calculating calibration measures
--------------------------------

.. autoclass:: CalibrationMeasures
    :members:

"""


import numpy as np
import math
from scipy.special import gamma

# from metrics.pairwise_measures import CacheFunctionOutput
from MetricsReloaded.utility.utils import (
    CacheFunctionOutput,
    max_x_at_y_more,
    max_x_at_y_less,
    min_x_at_y_more,
    min_x_at_y_less,
    trapezoidal_integration,
    one_hot_encode,
    median_heuristic
)


__all__ = [
    "CalibrationMeasures",
]


class CalibrationMeasures(object):
    def __init__(
        self,
        pred_proba,
        ref,
        case=None,
        measures=[],
        empty=False,
        dict_args={},
    ):
        self.measures_dict = {
            "ece": (self.expectation_calibration_error, "ECE"),
            "bs": (self.brier_score, "BS"),
            "rbs": (self.root_brier_score, "RBS"),
            "ls": (self.logarithmic_score, "LS"),
            "cwece": (self.class_wise_expectation_calibration_error, "cwECE"),
            "ece_kde": (self.kernel_based_ece, "ECE-KDE"),
            "kce":(self.kernel_calibration_error, "KCE"),
            "nll":(self.negative_log_likelihood,"NLL")
        }

        self.pred = np.asarray(pred_proba)
        self.ref = np.asarray(ref)
        self.n_classes = self.pred.shape[1]
        self.one_hot_ref = one_hot_encode(ref, self.n_classes)
        self.case = case
        self.flag_empty = empty
        self.dict_args = dict_args
        self.measures = measures if measures is not None else self.measures_dict

    def class_wise_expectation_calibration_error(self):
        r"""
        Class_wise version of the expectation calibration error

        Ananya Kumar, Percy S Liang, and Tengyu Ma. 2019. Verified uncertainty calibration. Advances in Neural Information
        Processing Systems 32 (2019).

        .. math::

            cwECE = \dfrac{1}{K}\sum_{k=1}^{K}\sum_{i=1}^{N}\dfrac{\vert B_{i,k} \vert}{N} \left(y_{k}(B_{i,k}) - p_{k}(B_{i,k})\right)

        :return: cwece
        """

        if "bins_ece" in self.dict_args:
            nbins = self.dict_args["bins_ece"]
        else:
            nbins = 10
        step = 1.0 / nbins
        range_values = np.arange(0, 1.00001, step)
        list_values = []
        numb_samples = self.pred.shape[0]
        class_pred = np.argmax(self.pred, 1)
        n_classes = self.pred.shape[1]
        for k in range(n_classes):
            list_values_k = []
            for (l, u) in zip(range_values[:-1], range_values[1:]):
                pred_k = self.pred[:, k]
                ref_tmp = np.where(
                    np.logical_and(pred_k > l, pred_k <= u),
                    self.ref,
                    np.ones_like(self.ref) * -1,
                )
                ref_sel = ref_tmp[ref_tmp > -1]
                ref_selk = np.where(
                    ref_sel == k, np.ones_like(ref_sel), np.zeros_like(ref_sel)
                )
                nsamples = np.size(ref_sel)
                prop = np.sum(ref_selk) / nsamples
                pred_tmp = np.where(
                    np.logical_and(pred_k > l, pred_k <= u),
                    pred_k,
                    np.ones_like(pred_k) * -1,
                )
                pred_sel = pred_tmp[pred_tmp > -1]
                if nsamples == 0:
                    list_values_k.append(0)
                else:
                    list_values_k.append(nsamples * np.abs(prop - np.mean(pred_sel)))

            list_values.append(np.sum(np.asarray(list_values_k)) / numb_samples)
        cwece = np.sum(np.asarray(list_values)) / n_classes
        return cwece

    def expectation_calibration_error(self):
        """
        Derives the expectation calibration error in the case of binary task
        bins_ece is the key in the dictionary for the number of bins to consider
        Default is 10

        .. math::

            ECE = \sum_{m=1}^{M} \dfrac{|B_m|}{n}(\dfrac{1}{|B_m|}\sum_{i \in B_m}1(pred_ik==ref_ik)-\dfrac{1}{|B_m|}\sum_{i \in B_m}pred_i)

        :return: ece

        """
        if "bins_ece" in self.dict_args:
            nbins = self.dict_args["bins_ece"]
        else:
            nbins = 10
        step = 1.0 / nbins
        range_values = np.arange(0, 1.00001, step)
        list_values = []
        numb_samples = 0
        pred_prob = self.pred[:,1]
        for (l, u) in zip(range_values[:-1], range_values[1:]):
            ref_tmp = np.where(
                np.logical_and(pred_prob > l, pred_prob <= u),
                self.ref,
                np.ones_like(self.ref) * -1,
            )
            ref_sel = ref_tmp[ref_tmp > -1]
            nsamples = np.size(ref_sel)
            prop = np.sum(ref_sel) / nsamples
            pred_tmp = np.where(
                np.logical_and(pred_prob > l, pred_prob <= u),
                pred_prob,
                np.ones_like(pred_prob) * -1,
            )
            pred_sel = pred_tmp[pred_tmp > -1]
            if nsamples == 0:
                list_values.append(0)
            else:
                list_values.append(nsamples * np.abs(prop - np.mean(pred_sel)))
            numb_samples += nsamples
        return np.sum(np.asarray(list_values)) / numb_samples

    def brier_score(self):
        """
        Calculation of the Brier score https://en.wikipedia.org/wiki/Brier_score
        here considering prediction probabilities as a vector of dimension N samples

        Glenn W Brier et al. 1950. Verification of forecasts expressed in terms of probability. Monthly weather review 78, 1
        (1950), 1–3.

        .. math::

            BS = \dfrac{1}{N}\sum_{i=1}{N}\sum_{j=1}^{C}(p_{ic}-r_{ic})^2

        where :math: `p_{ic}` is the probability for class c and :math: `r_{ic}` the binary reference for class c and element i

        :return: brier score (BS)

        """
        bs = np.mean(np.sum(np.square(self.one_hot_ref - self.pred),1))
        return bs

    def root_brier_score(self):
        """
        Determines the root brier score

        Gruber S. and Buettner F., Better Uncertainty Calibration via Proper Scores
        for Classification and Beyond, In Proceedings of the 36th International
        Conference on  Neural Information Processing Systems, 2022

        .. math::

            RBS = \sqrt{BS}

        :return: rbs
        """
        rbs = np.sqrt(self.brier_score())
        return rbs

    def logarithmic_score(self):
        """
        Calculation of the logarithmic score https://en.wikipedia.org/wiki/Scoring_rule
        
        .. math::

            LS = 1/N\sum_{i=1}^{N}\log{pred_ik}ref_{ik}

        :return: ls
        """
        eps = 1e-10
        log_pred = np.log(self.pred + eps)
        to_log = self.pred[np.arange(log_pred.shape[0]),self.ref]
        to_sum = log_pred[np.arange(log_pred.shape[0]),self.ref]
        ls =  np.mean(to_sum)
        return ls

    def distance_ij(self,i,j):
        """
        Determines the euclidean distance between two vectors of prediction for two samples i and j

        :return: distance
        """
        pred_i = self.pred[i,:]
        pred_j = self.pred[j,:]
        distance = np.sqrt(np.sum(np.square(pred_i - pred_j)))
        return distance


    def kernel_calculation(self, i,j):
        """
        Defines the kernel value for two samples i and j with the following definition for k(x_i,x_j)

        .. math::

            k(x_i,x_j) = exp(-||x_i-y_j||/ \\nu)I_{N}

        where :math: `\\nu` is the bandwith defined as the median heuristic if not specified in the options and N the number of classes

        :return: kernel_value

        """
        distance = self.distance_ij(i,j)
        if 'bandwidth_kce' in self.dict_args.keys():
            bandwidth = self.dict_args['bandwidth_kce']
        else:
            bandwidth = median_heuristic(self.pred)
        value = np.exp(-distance/bandwidth)
        identity = np.eye(self.pred.shape[1])
        kernel_value = value*identity
        return kernel_value

    def kernel_calibration_error(self):
        """
        Based on the paper Widmann, D., Lindsten, F., and Zachariah, D.
        Calibration tests in multi-class classification: A unifying framework.
        Advances in Neural Information Processing Systems, 32:12257–12267, 2019.

        :return: kce

        """
        one_hot_ref = one_hot_encode(self.ref, self.pred.shape[1])
        numb_samples = self.pred.shape[0]
        sum_tot = 0
        for i in range(0,numb_samples):
            for j in range(0,i):
                kernel = self.kernel_calculation(i,j)
                vect_i = one_hot_ref[i,:] - self.pred[i,:]
                vect_j = one_hot_ref[j,:] - self.pred[j,:]
                value_ij = np.matmul(vect_i, np.matmul(kernel,vect_j.T))
                sum_tot += value_ij
        multiplicative_factor = math.factorial(numb_samples)/ (2 * math.factorial(numb_samples-2))
        kce = 1/multiplicative_factor * sum_tot
        return kce



    def top_label_classification_error(self):
        """
        Calculation of the top-label classification error. Assumes pred_proba a matrix K x Numb observations
        with probability to be in class k for observation i in position (k,i)

        :return: tce

        """
        class_max = np.argmax(self.pred, 1)
        prob_pred_max = np.max(self.pred, 1)
        nclasses = self.pred.shape[1]
        numb_samples = self.pred.shape[0]
        prob = np.zeros([nclasses])
        prob_ref_values, prob_ref_counts = np.unique(self.ref, return_counts=True)
        for k in range(nclasses):
            idx = np.where(prob_ref_values == k)
            if len(idx) == 0:
                prob[k] = 0
            else:
                prob[k] = prob_ref_counts[idx[0]] / numb_samples

        prob_expected_max = prob[class_max]
        tce = np.sqrt(np.mean(np.square(prob_expected_max - prob_pred_max)))
        return tce

    def kernel_based_ece(self):
        """
        Calculates kernel based ECE

        Teodora Popordanoska, Raphael Sayer, and Matthew B Blaschko. 2022. A Consistent and Differentiable Lp Canonical
        Calibration Error Estimator. In Advances in Neural Information Processing Systems.

        .. math::
 
            ECE\_KDE = 1/N \sum_{j=1}^{N}||\dfrac{\sum_{i \\neq j}k_{Dir}(pred_j,pred_i)ref_i}{\sum_{i \\neq j}k_{Dir}(pred_j,pred_i)} - pred_j || 

        :return: ece_kde

        """
        ece_kde = 0
        one_hot_ref = one_hot_encode(self.ref, self.pred.shape[1])
        nclasses = self.pred.shape[1]
        numb_samples = self.pred.shape[0]
        norm_list = []
        for j in range(numb_samples):
            new_list = []
            new_vect = np.zeros([nclasses])
            for i in range(numb_samples):
                if j != i:
                    new_dir = self.dirichlet_kernel(j, i)
                    new_list.append(new_dir)
                    ref_tmp = one_hot_ref[i, :]
                    new_add = ref_tmp * new_dir
                    new_vect += new_add
            norm = np.sum(np.asarray(new_list))
            final_vect = new_vect / norm
            norm_list.append(final_vect - self.pred[j, :])

        full_array = np.vstack(norm_list)
        ece_kde = np.mean(np.sqrt(np.sum(np.square(full_array), 1)))

        return ece_kde

    def gamma_ik(self, i, k):
        """
        Definition of gamma value for sample i class k of the predictions

        .. math::

            gamma_{ik} = \Gamma(pred_{ik}/h + 1)

        where h is the bandwidth value set as default to 0.5

        :return gamma_ik

        """
        pred_ik = self.pred[i, k]
        if "bandwidth" in self.dict_args.keys():
            h = self.dict_args["bandwidth"]
        else:
            h = 0.5
        alpha_ik = pred_ik / h + 1
        gamma_ik = gamma(alpha_ik)
        return gamma_ik

    def dirichlet_kernel(self, j, i):
        """
        Calculation of Dirichlet kernel value for predictions of samples i and j

        .. math::

            k_{Dir}(x_j,x_i) = \dfrac{\Gamma(\sum_{k=1}^{K}\\alpha_{ik})}{\prod_{k=1}^{K}\\alpha_{ik}}\prod_{k=1}^{K}x_jk^{\\alpha_{ik}-1}
        
        :return: kernel_value

        """
        pred_i = self.pred[i, :]
        pred_j = self.pred[j, :]
        nclasses = self.pred.shape[1]
        if "bandwidth" in self.dict_args.keys():
            h = self.dict_args["bandwidth"]
        else:
            h = 0.5
        alpha_i = pred_i / h + 1
        numerator = gamma(np.sum(alpha_i))
        denominator = np.prod(gamma(alpha_i))
        prod = 1
        for k in range(nclasses):
            prod *= np.power(pred_j[k], alpha_i[k] - 1)
        kernel_value = numerator / denominator * prod
        return kernel_value


    def negative_log_likelihood(self):
        """
        Derives the negative log-likelihood defined as

        George Cybenko, Dianne P O’Leary, and Jorma Rissanen. 1998. The Mathematics of Information Coding, Extraction
        and Distribution. Vol. 107. Springer Science & Business Media.

        .. math::

            NLL = -\sum_{i=1}{N} log(p_{i,k} | y_i=k)

        :return: NLL

        """
        log_pred = np.log(self.pred)
        numb_samples = self.pred.shape[0]
        ll = np.sum(log_pred[range(numb_samples), self.ref])
        nll = -1 * ll
        return nll

    def to_dict_meas(self, fmt="{:.4f}"):
        """Given the selected metrics provides a dictionary with relevant metrics"""
        result_dict = {}
        for key in self.measures:
            result = self.measures_dict[key][0]()
            #result_dict[key] = fmt.format(result)
            result_dict[key] = result
        return result_dict 
