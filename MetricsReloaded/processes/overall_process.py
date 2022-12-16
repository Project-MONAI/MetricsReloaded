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
Overall process - :mod:`MetricsReloaded.processes.overall_process`
====================================================================

This module provides class to perform the :ref:`overall evaluation process  <processeval>`.

.. _processeval:

Performing the process associated with instance segmentation
------------------------------------

.. autoclass:: ProcessEvaluation
    :members:

"""


from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
from MetricsReloaded.processes.mixed_measures_processes import *
import warnings
from MetricsReloaded.utility.utils import combine_df
import pandas as pd

__all__ = [
    "ProcessEvaluation",
]

MAX = 1000

WORSE = {
    "ap": 0,
    "auroc": 0,
    "froc": MAX,
    "sens@spec": 0,
    "sens@ppv": 0,
    "spec@sens": 0,
    "fppi@sens": MAX,
    "ppv@sens": 0,
    "sens@fppi": 0,
    "fbeta": 0,
    "accuracy": 0,
    "balanced_accuracy": 0,
    "lr+": 0,
    "youden_ind": -1,
    "mcc": 0,
    "wck": -1,
    "cohens_kappa": -1,
    "iou": 0,
    "dsc": 0,
    "centreline_dsc": 0,
    "masd": MAX,
    "assd": MAX,
    "hd_perc": MAX,
    "hd": MAX,
    "boundary_iou": 0,
    "nsd": 0,
}

class ProcessEvaluation(object):
    """
    Performs the evaluation of the data stored in a pickled file according to all the measures, categories and choices of processing
    """

    def __init__(
        self,
        data,
        category,
        measures_pcc=[],
        measures_mcc=[],
        measures_boundary=[],
        measures_overlap=[],
        measures_mt=[],
        measures_detseg=[],
        localization='mask_iou',
        assignment='greedy_matching',
        flag_map=False,
        file=[],
        thresh_ass=0.5,
        case=True,
        flag_fp_in=True
    ):
        self.data = data
        self.category = category
        self.assignment = assignment
        self.localization = localization
        self.measures_overlap = measures_overlap
        self.measures_boundary = measures_boundary
        self.measures_mt = measures_mt
        self.measures_mcc = measures_mcc
        self.measures_pcc = measures_pcc
        self.measures_detseg = measures_detseg

        self.flag_map = flag_map
        self.thresh_ass = thresh_ass
        self.case = case
        self.flag_fp_in = flag_fp_in
        self.resdet, self.resseg, self.resmt, self.resmcc, self.rescal = self.process_data()


    def process_data(self):
        data = self.data
        df_resdet = None
        df_resseg = None
        df_resmt = None
        df_resmcc = None
        df_rescal = None
        if self.category == "Instance Segmentation":
            MLLS = MultiLabelLocSegPairwiseMeasure(
                pred_loc=data["pred_loc"],
                ref_loc=data["ref_loc"],
                pred_prob=data["pred_prob"],
                ref_class=data["ref_class"],
                pred_class=data["pred_class"],
                file=data["file"],
                flag_map=self.flag_map,
                assignment=self.assignment,
                localization=self.localization,
                measures_mt=self.measures_mt,
                measures_pcc=self.measures_pcc,
                measures_overlap=self.measures_overlap,
                measures_boundary=self.measures_boundary,
                measures_detseg=self.measures_detseg,
                thresh=self.thresh_ass,
                list_values=data["list_values"],
                per_case=self.case,
                flag_fp_in=self.flag_fp_in,
            )
            df_resseg, df_resdet, df_resmt = MLLS.per_label_dict()
        elif self.category == "Object Detection":
            MLDT = MultiLabelLocMeasures(
                pred_loc=data["pred_loc"],
                ref_loc=data["ref_loc"],
                pred_prob=data["pred_prob"],
                ref_class=data["ref_class"],
                pred_class=data["pred_class"],
                list_values=data["list_values"],
                localization=self.localization,
                assignment=self.assignment,
                thresh=self.thresh_ass,
                measures_pcc=self.measures_pcc,
                measures_mt=self.measures_mt,
                per_case=self.case,
                flag_fp_in=self.flag_fp_in,
            )
            df_resdet, df_resmt = MLDT.per_label_dict()
            df_resseg = None
        elif self.category in ["Image Classification", "Semantic Segmentation"]:
            MLPM = MultiLabelPairwiseMeasures(
                data["pred_class"],
                data["ref_class"],
                data["pred_prob"],
                measures_pcc=self.measures_pcc,
                measures_overlap=self.measures_overlap,
                measures_boundary=self.measures_boundary,
                measures_mcc=self.measures_mcc,
                measures_mt=self.measures_mt,
                list_values=data["list_values"],
                per_case=self.case,
            )
            df_bin, df_mt = MLPM.per_label_dict()
            df_mcc, df_cal = MLPM.multi_label_res()
            if self.category == "Image Classification":
                df_resdet = df_bin
                df_resseg = None
                df_resmt = df_mt
                df_resmcc = df_mcc
                df_rescal = df_cal
            else:
                df_resdet = None
                df_resseg = df_bin
                df_resmt = df_mt
                df_resmcc = df_mcc
        return df_resdet, df_resseg, df_resmt, df_resmcc, df_rescal

    def complete_missing_cases(self):
        if len(self.ref_missing) == 0:
            return
        if self.flag_ignore_missing:
            warnings.warn("The set up currently ignores any missing case / dataset")
            return 
        else:
            list_missing_det = []
            list_missing_seg = []
            list_missing_mt = []
            list_missing_mcc = []
            
            if self.case:
                for (i,f) in enumerate(self.ref_missing):
                    dict_mt = {}
                    dict_mcc = {}
                    dict_seg = {}
                    dict_det = {}
                    dict_mcc['case'] = i
                    for m in self.measures_mcc:
                        dict_mcc[m] = WORSE[m]
                    list_missing_mcc.append(dict_mcc)    
                    for l in self.list_values:
                        dict_seg = {}
                        dict_mt = {}
                        dict_det = {}
                        dict_seg['case'] = i
                        dict_det['case'] = i
                        dict_mt['case'] = i
                        dict_seg["label"] = l
                        dict_det["label"] = l
                        dict_mt["label"] = l
                        for m in self.measures_boundary:
                            dict_seg[m] = WORSE[m]
                        for m in self.measures_overlap:
                            dict_seg[m] = WORSE[m]
                        for m in self.measures_pcc:
                            dict_det[m] = WORSE[m]
                        for m in self.measures_mt:
                            dict_mt[m] = WORSE[m]
                        for m in self.measures_detseg:
                            dict_seg[m] = WORSE[m]
                        list_missing_seg.append(dict_seg)
                        list_missing_det.append(dict_det)
                        list_missing_mt.append(dict_mt)
            df_miss_det = pd.DataFrame.from_dict(list_missing_det)
            df_miss_seg = pd.DataFrame.from_dict(list_missing_seg)
            df_miss_mcc = pd.DataFrame.from_dict(list_missing_mcc)
            df_miss_mt = pd.DataFrame.from_dict(list_missing_mt)
            
            self.resdet = combine_df(self.resdet, df_miss_det)
            self.resseg = combine_df(self.resseg, df_miss_seg)
            self.resmt = combine_df(self.resmt, df_miss_mt)
            self.resmcc = combine_df(self.resmcc, df_miss_mcc)


