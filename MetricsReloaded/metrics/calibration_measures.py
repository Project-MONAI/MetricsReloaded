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
        ref_proba,
        case=None,
        measures=[],
        num_neighbors=8,
        pixdim=[1, 1, 1],
        empty=False,
        dict_args={},
    ):
        self.measures_dict = {
            "ece": (self.expectation_calibration_error, "ECE"),
            "bs": (self.brier_score, "BS"),
            "ls": (self.logarithmic_score, "LS"),
            "cwece": (self.class_wise_expectation_calibration_error, "cwECE"),
            "ece_kde": (self.kernel_based_ece, "ECE-KDE"),
        }

        self.pred = pred_proba
        self.ref = ref_proba
        self.case = case
        self.flag_empty = empty
        self.dict_args = dict_args
        self.measures = measures if measures is not None else self.measures_dict

    def class_wise_expectation_calibration_error(self):
        r"""
        Class_wise version of the expectation calibration error

        .. math::

            cwECE = \dfrac{1}{K}\sum_{k=1}^{K}\sum_{i=1}^{N}\dfrac{\vert B_{i,k} \vert}{N} \left(y_{k}(B_{i,k}) - p_{k}(B_{i,k})\right)


        """

        if "bins_ece" in self.dict_args:
            nbins = self.dict_args["bins_ece"]
        else:
            nbins = 10
        step = 1.0 / nbins
        range_values = np.arange(0, 1.00001, step)
        print(range_values)
        list_values = []
        numb_samples = self.pred.shape[0]
        class_pred = np.argmax(self.pred, 1)
        nclasses = self.pred.shape[1]
        for k in range(nclasses):
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

            print(list_values, numb_samples)
            list_values.append(np.sum(np.asarray(list_values_k)) / numb_samples)
        print(list_values)
        cwece = np.sum(np.asarray(list_values)) / nclasses
        return cwece

    def expectation_calibration_error(self):
        """
        Derives the expectation calibration error in the case of binary task
        bins_ece is the key in the dictionary for the number of bins to consider
        Default is 10

        """
        if "bins_ece" in self.dict_args:
            nbins = self.dict_args["bins_ece"]
        else:
            nbins = 10
        step = 1.0 / nbins
        range_values = np.arange(0, 1.00001, step)
        print(range_values)
        list_values = []
        numb_samples = 0
        for (l, u) in zip(range_values[:-1], range_values[1:]):
            ref_tmp = np.where(
                np.logical_and(self.pred > l, self.pred <= u),
                self.ref,
                np.ones_like(self.ref) * -1,
            )
            ref_sel = ref_tmp[ref_tmp > -1]
            nsamples = np.size(ref_sel)
            prop = np.sum(ref_sel) / nsamples
            pred_tmp = np.where(
                np.logical_and(self.pred > l, self.pred <= u),
                self.pred,
                np.ones_like(self.pred) * -1,
            )
            pred_sel = pred_tmp[pred_tmp > -1]
            if nsamples == 0:
                list_values.append(0)
            else:
                list_values.append(nsamples * np.abs(prop - np.mean(pred_sel)))
            numb_samples += nsamples
        print(list_values, numb_samples)
        return np.sum(np.asarray(list_values)) / numb_samples

    def brier_score(self):
        """
        Calculation of the Brier score https://en.wikipedia.org/wiki/Brier_score
        here considering prediction probabilities as a vector of dimension N samples
        """
        bs = np.mean(np.square(self.ref - self.pred))
        return bs

    def logarithmic_score(self):
        """
        Calculation of the logarithmic score https://en.wikipedia.org/wiki/Scoring_rule
        """
        eps = 1e-10
        log_pred = np.log(self.pred + eps)
        log_1pred = np.log(1 - self.pred + eps)
        print(log_pred, log_1pred, self.ref, 1 - self.ref)
        overall = self.ref * log_pred + (1 - self.ref) * log_1pred
        print(overall)
        ls = np.mean(overall)
        print(ls)
        return ls

    def distance_ij(self,i,j):
        pred_i = self.pred[i,:]
        pred_j = self.pred[j,:]
        distance = np.sqrt(np.sum(np.square(pred_i - pred_j)))
        return distance


    def kernel_calculation(self, i,j):
        distance = self.distance_ij(i,j)
        if 'bandwidth_kce' in self.dict_args.keys():
            bandwidth = self.dict_args['bandwidth_kce']
        else:
            bandwidth = median_heuristic(self.pred)
        value = np.exp(-distance/bandwidth)
        identity = np.ones([self.pred.shape[1], self.pred.shape[1]])
        return value * identity

    def kernel_calibration_error(self):
        one_hot_ref = one_hot_encode(self.ref)
        numb_samples = self.pred.shape[0]
        sum_tot = 0
        for i in range(0,numb_samples):
            for j in range(0,i):
                kernel = self.kernel_calculation(i,j)
                vect_i = one_hot_ref[i,:] - self.pred[i,:]
                vect_j = one_hot_ref[j,:] - self.pred[j,:]
                value_ij = np.matmul(vect_i, np.matmul(kernel,vect_j.T))
                sum_tot += value_ij
        multiplicative_factor = np.math.factorial(numb_samples)/ (2 * np.math.factorial(numb_samples-2))
        kce = 1/multiplicative_factor * sum_tot
        return kce



    def top_label_classification_error(self):
        """
        Calculation of the top-label classification error. Assumes pred_proba a matrix K x Numb observations
        with probability to be in class k for observation i in position (k,i)
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
        print(prob, prob_ref_counts, prob_expected_max, prob_pred_max)
        print(np.square(prob_expected_max - prob_pred_max))
        tce = np.sqrt(np.mean(np.square(prob_expected_max - prob_pred_max)))
        return tce

    def kernel_based_ece(self):
        ece_kde = 0
        one_hot_ref = one_hot_encode(self.ref, self.pred.shape[1])
        nclasses = self.pred.shape[1]
        numb_samples = self.pred.shape[0]
        print(nclasses, one_hot_ref)
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
                    print(new_add)
                    new_vect += new_add
            norm = np.sum(np.asarray(new_list))
            final_vect = new_vect / norm
            norm_list.append(final_vect - self.pred[j, :])

        full_array = np.vstack(norm_list)
        print(full_array.shape)
        ece_kde = np.mean(np.sqrt(np.sum(np.square(full_array), 1)))

        return ece_kde

    def gamma_ik(self, i, k):
        pred_ik = self.pred[i, k]
        if "bandwidth" in self.dict_args.keys():
            h = self.dict_args["bandwidth"]
        else:
            h = 0.5
        alpha_ik = pred_ik / h + 1
        gamma_ik = gamma(alpha_ik)
        return gamma_ik

    def dirichlet_kernel(self, j, i):
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

    def class_wise_brier_score(self):
        cwbs = 0
        return cwbs

    def kernel_calibration_error(self):
        """
        Based on the paper Widmann, D., Lindsten, F., and Zachariah, D.
        Calibration tests in multi-class classification: A unifying framework.
        Advances in Neural Information Processing Systems, 32:12257–12267, 2019.
        """
        kce = 0
        return kce

    def negative_log_likelihood(self):
        """
        Derives the negative log-likelihood defined as

        .. math::

            -\sum_{i=1}{N} log(p_{i,k} | y_i=k)

        """
        log_pred = np.log(self.pred)
        numb_samples = self.pred.shape[0]
        ll = np.sum(log_pred[range(numb_samples), self.ref])
        nll = -1 * ll
        return nll

    def root_brier_score(self):
        """
        Gruber S. and Buettner F., Better Uncertainty Calibration via Proper Scores
        for Classification and Beyond, In Proceedings of the 36th International
        Conference on  Neural Information Processing Systems, 2022
        """
        rbs = 0
        return rbs
