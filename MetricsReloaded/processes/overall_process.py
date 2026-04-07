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

.. autoclass:: ProcessEvaluation
    :members:

The different categories of task considered are:

* ImLC - Image Level Classification
* SemS - Semantic Segmentation
* ObD - Object detection
* InS - Instance segmentation

For each of these tasks only certain metrics are available and suitable. Error messages will be given and the processing interrupted if the chosen
task and the chosen evaluation measures are not compatible. 
Evaluation measures are classified into the following categories:

* Per class counting measures - measures_pcc
* Multi class counting measures - measures_mcc
* Overlap measures - measures_overlap
* Boundary measures - measures_boundary
* Multi threshold measures - measures_mt
* Calibration measures - measures_cal
* Combined detection and segmentation metrics - measures_detseg

The available measures per task are:

* ImLC:

  * multi threshold measures:

    * auroc - Area under the Receiver Operator Curve
    * ap - Average Precision
    * sens@spec - Sensitivity at Specificity
    * spec@sens - Specificity at Sensitivity
    * ppv@sens - Positive Predictive value at sensitivity

  * per class counting measures:

    * fbeta - FBeta score
    * lr+ - positive likelihood ratio
    * accuracy
    * ba - balance accuracy
    * ec - expected cost
    * nb - net benefit
    * numb_ref - number in reference
    * numb_pred - number in prediction
    * numb_tp - number of true positives
    * numb_fp - number of false positives
    * numb_fn - number of false negatives
    * cohens_kappa

  * multi class counting measures:

    * mcc - matthews correlation coefficient
    * wck - weighted cohen's kappa
    * ec - expected cost
    
  * calibration measures:

    * ls - logarithmic score
    * bs - Brier Score
    * cwece - Class-wise expectation calibration error
    * nll - Negative log-likelihood
    * rbs - Root Brier Score
    * ece_kde - Expectation Calibration Error with Kernel density estimation
    * kce - Kernel Calibration error
    * ece - Expectation Calibration Error

* Object Detection - ObD:

  * per class counting measures:

    * fbeta - FBeta score
    * numb_pred - number of predicted elements
    * numb_tp - number of true positives
    * numb_fp - number of false positives
    * numb_fn - number of false negatives
    * numb_ref - number of reference elements
    * sensitivity - sensitivity

  * multi-threshold measures:

    * sens@spec - sensitivity at specificity
    * spec@sens - specificity at sensitivity
    * sens@ppv - sensitivity at positive predictive value
    * ppv@sens - positive predictive value at sensitivity
    * sens@fppi - sensitivity at false positive per image
    * fppi@sens - false positive per image at sensitivity
    * ap - average precision
    * froc - free receiver operator curve

* Semantic segmentation - SemS:

  * per class measures of overlap: 
  
    * dsc - dice similarity coefficient
    * fbeta - FBeta score
    * cldice - centreline dice
    * iou - intersection over union
    
  * measures of boundary quality: 

    * assd - average symmetric surface distance
    * masd - mean average surface distance
    * hd - hausdorff distance
    * hd_perc - percentile of hausdorff distance
    * nsd - normalised surface dice
    * boundary_iou - boundary intersection over union
    
  * per class counting :

    * numb_ref - number of reference elements
    * numb_pred - number of predicted elements
    * numb_tp - number of true positives
    * numb_fp - number of false positives
    * numb_fn - number of false negatives

* Instance segmentation - InS:

  * combined measures of detection and segmentation

    * pq - panoptic quality

  * per class counting measures:

    * fbeta - FBeta score
    * numb_ref - number of reference instances
    * numb_pred - number of prediction instances
    * numb_tp - number of true positives
    * numb_fp - number of false positives
    * numb_fn - number of false negatives

  * multi-threshold measures:

    * sens@spec - sensitivity at specificity
    * spec@sens - specificity at sensitivity
    * sens@ppv - sensitivity at positive predictive value
    * ppv@sens - positive predictive value at sensitivity
    * fppi@sens - false positive per image at sensitivity
    * sens@fppi - sensitivity at false positive per image
    * ap - average precision
    * froc - free receiver operator curve

  * measures of overlap:

    * dsc - dice similarity coefficient
    * fbeta - fbeta score
    * cldice - centreline dice similarity coefficient
    * iou - intersection over union

  * measures of boundary quality:

    * hd - hausdorff distance
    * boundary_iou - boundary intersection over union
    * masd - mean average surface distance
    * assd - average symmetric surface distance
    * nsd - normalised surface dice
    * hd_perc - percentile of hausdorff distance
                  
 


"""


from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
from MetricsReloaded.processes.mixed_measures_processes import *
import warnings
from MetricsReloaded.utility.utils import combine_df, merge_list_df
import pandas as pd
import numpy as np

__all__ = [
    "ProcessEvaluation",
]

dict_valid={
    'ImLC': ['auroc','ap','sens@spec','spec@sens',
    'ppv@sens','fbeta','accuracy','ba',
    'ec','nb','mcc',
    'wck','lr+','bs','cwece',
    'nll','rbs','ece_kde','kce','ece',"numb_ref",'ls',
                    "numb_pred",
                    "numb_tp",
                    "numb_fp",
                    "numb_fn",]
,
    'ObD': ['fbeta','sens@spec','spec@sens','sens@ppv','ppv@sens','sens@fppi','fppi@sens','sensitivity','ap','froc', "numb_ref",
                    "numb_pred",
                    "numb_tp",
                    "numb_fp",
                    "numb_fn",
],
    'SemS': ['dsc','fbeta','cldice','iou','assd','masd','hd','hd_perc','nsd','boundary_iou',"numb_ref",
                    "numb_pred",
                    "numb_tp",
                    "numb_fp",
                    "numb_fn",
],
    'InS': ['pq','fbeta','sens@spec','spec@sens','sens@ppv','ppv@sens',
    'fppi@sens','sens@fppi','ap','froc','dsc','cldice','iou','hd','boundary_iou',
    'masd','assd','nsd','hd_perc',"numb_ref",
                    "numb_pred",
                    "numb_tp",
                    "numb_fp",
                    "numb_fn",]

}

MAX = 1000

LIST_DISTANCE = ['hd','masd','assd','hd_perc']

WORSE = {
    "ap": 0,
    "auroc": 0,
    "froc": 0,
    "sens@spec": 0,
    "sens@ppv": 0,
    "spec@sens": 0,
    "fppi@sens": MAX,
    "ppv@sens": 0,
    "sens@fppi": 0,
    "fbeta": 0,
    "ec":1,
    "accuracy": 0,
    "ba": 0,
    "lr+": 0,
    "youden_ind": -1,
    "mcc": 0,
    "wck": -1,
    "cohens_kappa": -1,
    "iou": 0,
    "dsc": 0,
    "cldice": 0,
    "masd": MAX,
    "assd": MAX,
    "hd_perc": MAX,
    "hd": MAX,
    "boundary_iou": 0,
    "nsd": 0,
}

BEST = {
    "ap": 1,
    "auroc": 1,
    "froc": 1,
    "sens@spec": 1,
    "sens@ppv": 1,
    "spec@sens": 1,
    "fppi@sens": 0,
    "ppv@sens": 1,
    "sens@fppi": 1,
    "fbeta": 1,
    "ec":0,
    "accuracy": 1,
    "ba": 1,
    "lr+": 1,
    "youden_ind": 1,
    "mcc": 1,
    "wck": 1,
    "cohens_kappa": 1,
    "iou": 1,
    "dsc": 1,
    "cldice": 1,
    "masd": 0,
    "assd": 0,
    "hd_perc": 0,
    "hd": 0,
    "boundary_iou": 1,
    "nsd": 1,
}

NAN_LIST = ["iou","dsc","fbeta","masd",'cldice','hd','hd_perc','assd','boundary_iou','nsd']

class ProcessEvaluation(object):
    """
    Performs the evaluation of the data stored in a pickled file according to all the measures, categories and choices of processing

    :param data: dictionary containing all the data to be used for the comparison; possible keys include "pred_loc", "ref_loc", "pred_prob", 
    :param category: task to be considered choice among ImLC, ObD, SemS, InS
    :param measures_pcc: list of per class counting measures (these need to be adequate for the chosen task category)
    :param measures_mcc: list of multi class counting measures
    :param measures_boundary: list of measures to assess boundary quality
    :param measures_overlap: list of measures to assess overlap quality
    :param measures_mt: list of multi-threshold measures
    :param measures_detseg: list of measures assessing jointly detection and segmentation performance
    :param measures_cal: list of calibration measures (only available for image level classification class)
    :param localization: choice for localization strategy (used in Instance segmentation and Object detection tasks)
    :param assignment: choice for the assignment strategy (used in Instance segmentation and Object detection tasks)
    :param pixdim: pixel dimensions as list
    :param flag_map: indication whether nifti images indicating true positive elements for the reference, the prediction and errors should be created (done only for instance segmentation)
    :param file: name of files
    :param thresh_ass: threshold chosen for the assignment (default 0.5)
    :param case: indication of the handling of cases separately (True) or jointly (False)
    :param flag_fp_in: indicates that false positive should be accounted for 
    :param ignore_missing: indicates whether the missing predictions should be considered in the overall assessment (True) or not (False)
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
        measures_cal=[],
        localization='mask_iou',
        assignment='greedy_matching',
        pixdim=[],
        flag_map=False,
        file=[],
        thresh_ass=0.5,
        case=True,
        flag_fp_in=True,
        ignore_missing = False
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
        self.measures_cal = measures_cal

        self.flag_map = flag_map
        self.thresh_ass = thresh_ass
        self.case = case
        self.flag_fp_in = flag_fp_in
        self.flag_ignore_missing = ignore_missing
        self.flag_valid = self.check_valid_measures_cat()
        self.pixdim = pixdim
        if self.flag_valid:
            self.process_data()
            if 'ref_missing' in self.data.keys():
                self.complete_missing_cases()
            if 'weights_labels' in self.data.keys():
                self.weights_labels = self.data['weights_labels']
            else:
                self.weights_labels = {}
                for v in self.data['list_values']:
                    self.weights_labels[v] = 1
            self.grouped_lab = self.label_aggregation()
            if self.case:
                self.get_stats_res()

    def check_valid_measures_cat(self):
        """
        Function checking whether the category and the combination of measures suggested are suitable for continuing the process

        :return: flag_valid
        """
        flag_valid = True
        if self.category not in ['ImLC','SemS','InS','ObD']:
            warnings.warn('No appropriate category chosen, please choose between ImLC, SemS, InS and ObD')
            return False
        all_measures = self.measures_boundary + self.measures_cal + self.measures_detseg + self.measures_mcc + self.measures_mt + self.measures_overlap + self.measures_pcc

        for k in all_measures:
            if k not in dict_valid[self.category]:
                warnings.warn( '%s is not a suitable metric for %s' %(k,self.category))
                flag_valid = False
        return flag_valid



    def process_data(self):
        """
        Performs the processing of the data according to the details given in the setting up of the process
        Contributes to the attribution of one dataframe per type of measures :

        * resdet - detection results
        * resseg - segmentation results
        * resmt - multi-threshold results
        * resmcc - multi class counting results
        * rescal - calibration results

        All these dataframes are initialised as None and replaced according to the chosen task. The tasks should yield the following outputs:
        
        * ImLC:

          * resdet
          * rescal
          * resmt
          * resmcc

        * SemS:

          * resseg

        * ObD:

          * resdet
          * resmt
          * resmcc

        * InS:
        
          * resdet
          * resseg
          * resmt
          * resmcc
        
        """
        data = self.data
        df_resdet = None
        df_resseg = None
        df_resmt = None
        df_resmcc = None
        df_rescal = None
        if self.category == "InS":
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
                pixdim=self.pixdim
            )
            df_resseg, df_resdet, df_resmt = MLLS.per_label_dict()
        elif self.category == "ObD":
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
                pixdim=self.pixdim
                
            )
            df_resdet, df_resmt = MLDT.per_label_dict()
            df_resseg = None
        elif self.category in ["ImLC", "SemS"]:
            if 'names' in data.keys():
                list_names=data['names']
            else:
                list_names = []
            MLPM = MultiLabelPairwiseMeasures(
                data["pred_class"],
                data["ref_class"],
                data["pred_prob"],
                measures_pcc=self.measures_pcc,
                measures_overlap=self.measures_overlap,
                measures_boundary=self.measures_boundary,
                measures_mcc=self.measures_mcc,
                measures_mt=self.measures_mt,
                measures_calibration=self.measures_cal,
                list_values=data["list_values"],
                names=list_names,
                per_case=self.case,
                pixdim=self.pixdim
            )
            df_bin, df_mt = MLPM.per_label_dict()
            df_mcc, df_cal = MLPM.multi_label_res()
            # print(df_bin, 'BIN')
            # print(df_mt, 'MT')
            # print(df_mcc, 'MCC'),
            # print(df_cal, 'CAL')
            if self.category == "ImLC":
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
        self.resdet = df_resdet
        self.resseg = df_resseg
        self.resmt = df_resmt
        self.resmcc = df_resmcc
        self.rescal = df_rescal
        self.create_mapping_column_nan_replaced_seg()
        return
    
    def create_mapping_column_nan_replaced_seg(self):
        """
        For each measure (segmentation) for which nan are possible 
        creates an additional column in which nans are replaced by value (worse or best according to situation
        """
        list_to_map = []
        for x in self.measures_boundary:
            if x in NAN_LIST:
                list_to_map.append(x)
        for x in self.measures_overlap:
            if x in NAN_LIST:
                list_to_map.append(x)
        for k in list_to_map:
            self.resseg[k+'_nanrep'] = self.resseg[k]
            
            self.resseg[k+'_nanrep'] = np.where(np.logical_and(self.resseg[k].isna(),self.resseg['check_empty']=='Both')
                                                               ,BEST[k],self.resseg[k+'_nanrep'])
            self.resseg[k+'_nanrep'] = np.where(np.logical_and(self.resseg[k+'_nanrep'].isna(),self.resseg['check_empty']=='Ref')
                                                               ,WORSE[k],self.resseg[k+'_nanrep'])
            self.resseg[k+'_nanrep'] = np.where(np.logical_and(self.resseg[k+'_nanrep'].isna(),self.resseg['check_empty']=='Pred')
                                                               ,WORSE[k],self.resseg[k+'_nanrep'])
            self.resseg[k+'_nanrep'] = np.where(np.logical_and(self.resseg[k].isna(),k in LIST_DISTANCE)
                                                               ,self.resseg['worse_dist'],self.resseg[k+'_nanrep'])

        return
        
        

    
    def identify_empty_ref(self):
        return

    def complete_missing_cases(self):
        if len(self.data['ref_missing']) == 0:
            return
        if self.flag_ignore_missing:
            warnings.warn("The set up currently ignores any missing case / dataset")
            return 
        else:
            list_missing_det = []
            list_missing_seg = []
            list_missing_mt = []
            list_missing_mcc = []
            numb_valid = len(self.data['ref_class'])
            if self.case:
                for (i,f) in enumerate(self.data['ref_missing']):
                    dict_mt = {}
                    dict_mcc = {}
                    dict_seg = {}
                    dict_det = {}
                    dict_mcc['case'] = i + numb_valid
                    for m in self.measures_mcc:
                        dict_mcc[m] = WORSE[m]
                    list_missing_mcc.append(dict_mcc)    
                    for l in self.data['list_values']:
                        dict_seg = {}
                        dict_mt = {}
                        dict_det = {}
                        
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
                        if len(self.measures_boundary) + len(self.measures_overlap) > 0:
                            dict_seg['case'] = i + numb_valid
                            dict_seg["label"] = l
                            list_missing_seg.append(dict_seg)
                        if len(self.measures_pcc) + len(self.measures_detseg) > 0 : 
                            dict_det['case'] = i + numb_valid
                            dict_det["label"] = l
                            list_missing_det.append(dict_det)
                        if len(self.measures_mt) > 0:
                            dict_mt['case'] = i + numb_valid
                            dict_mt["label"] = l
                            list_missing_mt.append(dict_mt)
            df_miss_det = pd.DataFrame.from_dict(list_missing_det)
            df_miss_seg = pd.DataFrame.from_dict(list_missing_seg)
            df_miss_mcc = pd.DataFrame.from_dict(list_missing_mcc)
            df_miss_mt = pd.DataFrame.from_dict(list_missing_mt)
            self.resdet = combine_df(self.resdet, df_miss_det)
            self.resseg = combine_df(self.resseg, df_miss_seg)
            self.resmt = combine_df(self.resmt, df_miss_mt)
            self.resmcc = combine_df(self.resmcc, df_miss_mcc)

    def label_aggregation(self, option='average',dict_args={}):
        if len(self.data['list_values']) == 1:
            # print('DET', self.resdet,'CAL',self.rescal, 'SEG',self.resseg,'MT', self.resmt,'MCC', self.resmcc)
            df_grouped_all = merge_list_df([self.resdet, self.resseg, self.resmt,self.resmcc, self.rescal])
            return df_grouped_all
        df_all_labels = merge_list_df([self.resdet, self.resseg, self.resmt], on=['label','case'])
        df_all_labels['weights_labels'] = 1
        df_all_labels['prevalence_labels'] = 1 
        for k in self.weights_labels.keys():
            df_all_labels['weights_labels'] = np.where(df_all_labels['label']==k,self.weights_labels[k],df_all_labels['weights_labels'])
        for (c,rc) in enumerate(self.data['ref_class']):
            values,counts = np.unique(rc, return_counts=True)
            for (v,co) in zip(values,counts):
                df_all_labels['prevalence_labels'] = np.where(np.logical_and(df_all_labels['case']==c, df_all_labels['label']==v),co,df_all_labels['prevalence_labels'])
        wm = lambda x: np.ma.average(np.ma.masked_array(x,np.isnan(x)), weights=df_all_labels.loc[x.index, "prevalence_labels"])
        wm2 = lambda x: np.ma.average(np.ma.masked_array(x,np.isnan(x)), weights=df_all_labels.loc[x.index, "weights_labels"])
        wm3 = lambda x: np.ma.average(np.ma.masked_array(x,np.isnan(x)))
        list_measures = self.measures_boundary + self.measures_overlap + self.measures_detseg + self.measures_pcc + self.measures_mt
        dict_measures = {k:[('prevalence',wm),('weights',wm2),('average',wm3)] for k in list_measures}
        df_grouped_lab = df_all_labels.groupby('case',as_index=False).agg(dict_measures).reset_index()
        df_grouped_lab.columns = ['_'.join(col).rstrip('_') for col in df_grouped_lab.columns.values
]
        
        # print(df_grouped_lab, " grouped lab ")                                             
        df_grouped_all = merge_list_df([df_grouped_lab.reset_index(), self.resmcc, self.rescal], on=['case'])
        # print(df_grouped_all, 'grouped all')
        return df_grouped_all

    def get_stats_res(self):
        df_stats_all = self.grouped_lab.describe()
        df_all_labels = merge_list_df([self.resdet, self.resseg, self.resmt], on=['label','case'])
        df_stats_lab = df_all_labels.groupby('label').describe()
        self.stats_lab = df_stats_lab
        self.stats_all = df_stats_all
        return 

    





