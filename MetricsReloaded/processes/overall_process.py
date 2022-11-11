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


__all__ = [
    'ProcessEvaluation',
]

class ProcessEvaluation(object):
    """
    Performs the evaluation of the data stored in a pickled file according to all the measures, categories and choices of processing
    """

    def __init__(
        self,
        data,
        category,
        localization,
        assignment,
        measures_pcc=[],
        measures_mcc=[],
        measures_boundary=[],
        measures_overlap=[],
        measures_mt=[],
        measures_detseg=[],
        flag_map=False,
        file=[],
        thresh_ass=0.5,
        case=False
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
        
        self.flag_map=flag_map
        self.thresh_ass=thresh_ass
        self.case=case

    def process_data(self):
        data = self.data
        df_resdet = None
        df_resseg = None
        df_resmt = None
        df_resmcc = None
        if self.category == "Instance Segmentation":
            MLLS = MultiLabelLocSegPairwiseMeasure(
                pred_loc=data["pred_loc"],
                ref_loc=data["ref_loc"],
                pred_prob=data["pred_prob"],
                ref_class=data["ref_class"],
                pred_class=data["pred_class"],
                file=data['file'],
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
                per_case=self.case
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
                per_case=self.case
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
                per_case=self.case
            )
            df_bin, df_mt = MLPM.per_label_dict()
            df_mcc = MLPM.multi_label_res()
            if self.category == "Image Classification":
                df_resdet = df_bin
                df_resseg = None
                df_resmt = df_mt
                df_resmcc = df_mcc
            else:
                df_resdet = None
                df_resseg = df_bin
                df_resmt = df_mt
                df_resmcc = df_mcc
        return df_resdet, df_resseg, df_resmt, df_resmcc