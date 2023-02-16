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
Mixed measures processes - :mod:`MetricsReloaded.processes.mixed_measures_processes`
===================================================================================

This module provides classes for performing the evaluation processes of
 :ref:`instance segmentation <instanceseg>`, :ref:`multi label instance segmentation <mlinstanceseg>`, 
:ref:`multilabel object detection <mlod>` and :ref:`multi class classification  <multiclass>`.

.. _instanceseg:

Performing the process associated with instance segmentation
------------------------------------------------------------

.. autoclass:: MixedLocSegPairwiseMeasure
    :members:

.. _mlinstanceseg:

Performing the process associated with multiple labels in instance segmentation
-------------------------------------------------------------------------------

.. autoclass:: MultiLabelLocSegPairwiseMeasure
    :members:

Performing the process associated with multiple labels in instance segmentation
-------------------------------------------------------------------------------

.. autoclass:: MultiLabelLocMeasures
    :members:

Performing the process associated with multiple labels in classification (semantic segmentation or image level classification)
------------------------------------------------------------------------------------------------------------------------------

.. autoclass:: MultiLabelPairwiseMeasures
    :members:
"""


from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures
from MetricsReloaded.metrics.pairwise_measures import (
    BinaryPairwiseMeasures,
    MultiClassPairwiseMeasures,
)
from MetricsReloaded.metrics.calibration_measures import CalibrationMeasures
from MetricsReloaded.utility.assignment_localization import AssignmentMapping
import numpy as np
import pandas as pd
import nibabel as nib
import os
import warnings


__all__ = [
    "MixedLocSegPairwiseMeasure",
    "MultiLabelLocSegPairwiseMeasure",
    "MultiLabelLocMeasures",
    "MultiLabelPairwiseMeasures",
]


class MixedLocSegPairwiseMeasure(object):
    def __init__(
        self,
        pred,
        ref,
        list_predimg,
        list_refimg,
        pred_prob,
        measures_overlap=[],
        measures_boundary=[],
        measures_mt=[],
        measures_pcc=[],
        measures_detseg=[],
        connectivity=1,
        pixdim=[1, 1, 1],
        empty=False,
        dict_args={},
    ):
        self.pred = pred
        self.ref = ref
        self.predimg = list_predimg
        self.refimg = list_refimg
        self.measures_mt = measures_mt
        self.measures_detseg = measures_detseg
        self.measures_det = measures_pcc
        self.measures_seg = measures_boundary + measures_overlap
        self.dict_args = dict_args
        self.prob_res = ProbabilityPairwiseMeasures(
            pred_prob, ref, self.measures_mt, self.dict_args
        )
        self.det_res = BinaryPairwiseMeasures(
            pred, ref, self.measures_det, dict_args=self.dict_args
        )
        self.seg_res = [
            BinaryPairwiseMeasures(p, r, self.measures_seg, dict_args=self.dict_args)
            for (p, r) in zip(list_predimg, list_refimg)
        ]

    def segmentation_quality(self):
        """
        Calculates the segmentation quality in an instance segmentation case with SQ defined as the average IoU over TP instances
        """
        list_iou = []
        for (p, r) in zip(self.predimg, self.refimg):
            PE = BinaryPairwiseMeasures(p, r)
            list_iou.append(PE.intersection_over_union())
        print(list_iou, " is list iou")
        return np.mean(np.asarray(list_iou))

    def detection_quality(self):
        """
        Calculates the detection quality for a case of instance segmentation defined as the F1 score
        """
        PE = BinaryPairwiseMeasures(self.pred, self.ref)
        #print("pred is ", self.pred, "ref is ", self.ref)
        return PE.fbeta()

    def panoptic_quality(self):
        """
        Calculates the panopitic quality defined as the production between
        detection quality and segmentation quality

        Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. 2019. Panoptic segmentation. In
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9404–9413.

        :return: PQ
        """
        print("DQ ", self.detection_quality())
        print("SQ ", self.segmentation_quality())
        DQ = self.detection_quality()
        SQ = self.segmentation_quality()
        if np.isnan(SQ):
            if DQ == 0:
                SQ = 0
            else:
                SQ = 1
                # TODO modify to nan if this is the value adopted for empty situations
        print("PQ is ", DQ * SQ, DQ, SQ)
        return DQ * SQ

    def to_dict_mt(self):
        dict_output = self.prob_res.to_dict_meas()
        return dict_output

    def to_dict_det(self):
        dict_output = self.det_res.to_dict_meas()
        if "PQ" in self.measures_detseg:
            dict_output["PQ"] = self.panoptic_quality()
        return dict_output

    def to_pd_seg(self):
        list_res = []
        for ps in self.seg_res:
            dict_tmp = ps.to_dict_meas()
            list_res.append(dict_tmp)
        return pd.DataFrame.from_dict(list_res)


class MultiLabelLocSegPairwiseMeasure(object):
    """
    This class represents the processing for instance segmentation on true positive 
    Characterised by the predicted classes and associated reference classes
    """
    
    def __init__(
        self,
        pred_class,
        ref_class,
        pred_loc,
        ref_loc,
        pred_prob,
        list_values,
        names=[],
        measures_pcc=[],
        measures_overlap=[],
        measures_boundary=[],
        measures_detseg=[],
        measures_mt=[],
        per_case=True,
        flag_map=True,
        file=[],
        connectivity=1,
        pixdim=[1, 1, 1],
        empty=False,
        assignment="Greedy_IoU",
        localization="iou",
        thresh=0.5,
        flag_fp_in=True,
        dict_args={},
    ):

   
        self.pred_loc = pred_loc
        self.list_values = list_values
        self.ref_class = ref_class
        self.ref_loc = ref_loc
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.measures_mt = measures_mt
        self.measures_pcc = measures_pcc
        self.measures_overlap = measures_overlap
        self.measures_boundary = measures_boundary
        self.measures_detseg = measures_detseg
        self.per_case = per_case
        self.assignment = assignment
        self.localization = localization
        self.matching = []
        self.file = file
        self.thresh = thresh
        self.dict_args = dict_args
        self.flag_map = flag_map
        self.names = names
        self.flag_fp_in = flag_fp_in
        if len(self.names) == 0:
            self.names = self.file
        if len(self.names) < len(self.ref_loc):
            self.names = range(len(self.ref_loc))
        self.flag_valid_prob = True
        if pred_prob is None or pred_prob[0] is None:
            self.flag_valid_prob = False

    def create_map(self, list_maps, file_ref, category):
        affine = nib.load(file_ref).affine
        data = nib.load(file_ref).get_fdata()
        final_class = np.zeros_like(data)
        for f in list_maps:
            final_class += f
        nib_img = nib.Nifti1Image(final_class, affine)
        path, name = os.path.split(file_ref)
        name_new = category + "_" + name
        name_fin = path + os.path.sep + name_new
        print(name_fin)
        nib.save(nib_img, name_fin)

    def per_label_dict(self):
        list_det = []
        list_seg = []
        list_mt = []
        #print(self.list_values)
        for lab in self.list_values:
            list_pred = []
            list_ref = []
            list_prob = []
            list_pred_loc = []
            list_ref_loc = []
            for (case, name) in zip(range(len(self.pred_class)), self.names):
                #print(self.pred_prob[case], self.pred_prob[case].shape)
                pred_class_case = np.asarray(self.pred_class[case])
                ref_class_case = np.asarray(self.ref_class[case])
                ind_pred = np.where(pred_class_case == lab)
                #print(ind_pred)
                pred_tmp = np.where(
                    pred_class_case == lab,
                    np.ones_like(pred_class_case),
                    np.zeros_like(pred_class_case),
                )
                ref_tmp = np.where(
                    ref_class_case == lab,
                    np.ones_like(ref_class_case),
                    np.zeros_like(ref_class_case),
                )
                ind_ref = np.where(ref_class_case == lab)
                pred_loc_tmp = [self.pred_loc[case][i] for i in ind_pred[0]]
                ref_loc_tmp = [self.ref_loc[case][i] for i in ind_ref[0]]
                if self.flag_valid_prob:
                    pred_prob_tmp = [self.pred_prob[case][i, lab] for i in ind_pred[0]]
                else:
                    pred_prob_tmp = None
                #print(len(pred_loc_tmp), len(ref_loc_tmp), lab, case)
                AS = AssignmentMapping(
                    pred_loc=pred_loc_tmp,
                    ref_loc=ref_loc_tmp,
                    pred_prob=pred_prob_tmp,
                    assignment=self.assignment,
                    localization=self.localization,
                    thresh=self.thresh,
                    flag_fp_in=self.flag_fp_in
                )

                pred_tmp_fin = np.asarray(AS.df_matching["pred"])
                pred_tmp_fin = np.where(
                    pred_tmp_fin > -1,
                    np.ones_like(pred_tmp_fin),
                    np.zeros_like(pred_tmp_fin),
                )
                ref_tmp_fin = np.asarray(AS.df_matching["ref"])
                pred_prob_fin = np.asarray(AS.df_matching["pred_prob"])
                ref_tmp_fin = np.where(
                    ref_tmp_fin > -1,
                    np.ones_like(ref_tmp_fin),
                    np.zeros_like(ref_tmp_fin),
                )
                (
                    pred_loc_tmp_fin,
                    ref_loc_tmp_fin,
                    pred_fp_loc,
                    ref_fn_loc,
                ) = AS.matching_ref_predseg()
                if self.flag_map and len(self.file) == len(self.pred_class):
                    self.create_map(pred_loc_tmp_fin, self.file[case], "TP_Pred")
                    self.create_map(ref_loc_tmp_fin, self.file[case], "TP_Ref")
                    self.create_map(pred_fp_loc, self.file[case], "FP")
                    self.create_map(ref_fn_loc, self.file[case], "FN")
                print("assignment done")
                if self.per_case:
                    # pred_loc_tmp_fin = pred_loc_tmp[list_valid]
                    # list_ref_valid = list(df_matching[df_matching['seg'].isin(list_valid)]['ref'])
                    # ref_loc_tmp_fin = ref_loc_tmp[list_ref_valid]

                    MLSPM = MixedLocSegPairwiseMeasure(
                        pred=pred_tmp_fin,
                        ref=ref_tmp_fin,
                        list_predimg=pred_loc_tmp_fin,
                        list_refimg=ref_loc_tmp_fin,
                        pred_prob=pred_prob_fin,
                        measures_detseg=self.measures_detseg,
                        measures_pcc=self.measures_pcc,
                        measures_boundary=self.measures_boundary,
                        measures_overlap=self.measures_overlap,
                        measures_mt=self.measures_mt,
                        dict_args=self.dict_args,
                    )
                    if len(self.measures_boundary) + len(self.measures_overlap)  > 0:
                        seg_res = MLSPM.to_pd_seg()
                        seg_res["label"] = lab
                        seg_res["case"] = name
                        list_seg.append(seg_res)
                    if len(self.measures_pcc) + len(self.measures_detseg)>0:
                        det_res = MLSPM.to_dict_det()
                        det_res["label"] = lab
                        det_res["case"] = name
                        list_det.append(det_res)
                    if len(self.measures_mt) > 0 :
                        mt_res = MLSPM.to_dict_mt()
                        mt_res["label"] = lab
                        mt_res["case"] = name
                        list_mt.append(mt_res)
                    df_matching = AS.df_matching
                    df_matching["case"] = name
                    df_matching["label"] = lab
                    self.matching.append(df_matching)
                else:
                    for p in pred_loc_tmp_fin:
                        list_pred_loc.append(p)
                    for r in ref_loc_tmp_fin:
                        list_ref_loc.append(r)
                    for p in pred_tmp_fin:
                        list_pred.append(p)
                    for r in ref_tmp_fin:
                        list_ref.append(r)
                    for p in pred_prob_tmp:
                        list_prob.append(p)
                    df_matching = AS.df_matching
                    df_matching["case"] = name
                    df_matching["label"] = lab
                    self.matching.append(df_matching)
            if not self.per_case:
                overall_pred = np.asarray(list_pred)
                overall_ref = np.asarray(list_ref)
                overall_prob = np.asarray(list_prob)
                MLSPM = MixedLocSegPairwiseMeasure(
                    pred=overall_pred,
                    ref=overall_ref,
                    list_predimg=list_pred_loc,
                    list_refimg=list_ref_loc,
                    pred_prob=overall_prob,
                    measures_detseg=self.measures_detseg,
                    measures_pcc=self.measures_pcc,
                    measures_boundary=self.measures_boundary,
                    measures_overlap=self.measures_overlap,
                    measures_mt=self.measures_mt,
                    dict_args=self.dict_args,
                )
                if len(self.measures_boundary) + len(self.measures_overlap) > 0 :
                    res_seg = MLSPM.to_pd_seg()
                    res_seg["label"] = lab
                    list_seg.append(res_seg)
                if len(self.measures_pcc) + len(self.measures_detseg) > 0:
                    res_det = MLSPM.to_dict_det()
                    res_det["label"] = lab
                    list_det.append(res_det)
                if len(self.measures_mt) > 0:
                    res_mt = MLSPM.to_dict_mt()
                    res_mt["label"] = lab
                    list_mt.append(res_mt)
        self.matching = pd.concat(self.matching)
        if len(list_seg) == 0:
            df_seg = None
        else:
            df_seg = pd.concat(list_seg)
        return (
            df_seg,
            pd.DataFrame.from_dict(list_det),
            pd.DataFrame.from_dict(list_mt),
        )


class MultiLabelLocMeasures(object):
    def __init__(
        self,
        pred_loc,
        ref_loc,
        pred_class,
        ref_class,
        pred_prob,
        list_values,
        names=[],
        measures_pcc=[],
        measures_mt=[],
        per_case=False,
        assignment="Greedy IoU",
        localization="iou",
        thresh=0.5,
        flag_fp_in=True,
        dict_args={},
    ):
        self.pred_loc = pred_loc
        self.ref_loc = ref_loc
        self.ref_class = ref_class
        self.pred_class = pred_class
        self.list_values = list_values
        self.pred_prob = pred_prob
        self.measures_pcc = measures_pcc
        self.measures_mt = measures_mt
        self.per_case = per_case
        self.assignment = assignment
        self.localization = localization
        self.thresh = thresh
        self.dict_args = {}
        self.names = names
        self.flag_fp_in = flag_fp_in
        if len(self.names) < len(self.ref):
            self.names = range(len(self.ref))
        self.flag_valid_proba = True
        if pred_prob is None or pred_prob[0] is None:
            self.flag_valid_proba=False

    def per_label_dict(self):
        list_det = []
        list_mt = []
        for lab in self.list_values:
            list_pred = []
            list_ref = []
            list_prob = []
            for (case, name) in zip(range(len(self.ref_class)), self.names):
                pred_arr = np.asarray(self.pred_class[case])
                ref_arr = np.asarray(self.ref_class[case])
                ind_pred = np.where(pred_arr == lab)
                pred_tmp = np.where(
                    pred_arr == lab, np.ones_like(pred_arr), np.zeros_like(pred_arr)
                )
                ref_tmp = np.where(
                    ref_arr == lab, np.ones_like(ref_arr), np.zeros_like(ref_arr)
                )
                ind_ref = np.where(ref_arr == lab)
                pred_loc_tmp = [self.pred_loc[case][f] for f in ind_pred[0]]
                ref_loc_tmp = [self.ref_loc[case][f] for f in ind_ref[0]]
                if self.flag_valid_proba:
                    pred_prob_tmp = [self.pred_prob[case][lab,f] for f in ind_pred[0]]
                else:
                    pred_prob_tmp = None
                AS = AssignmentMapping(
                    pred_loc=pred_loc_tmp,
                    ref_loc=ref_loc_tmp,
                    pred_prob=pred_prob_tmp,
                    assignment=self.assignment,
                    localization=self.localization,
                    thresh=self.thresh,
                    flag_fp_in=self.flag_fp_in
                )
                df_matching = AS.df_matching
                pred_tmp_fin = np.asarray(df_matching["pred"])
                pred_tmp_fin = np.where(
                    pred_tmp_fin > -1,
                    np.ones_like(pred_tmp_fin),
                    np.zeros_like(pred_tmp_fin),
                )
                ref_tmp_fin = np.asarray(df_matching["ref"])
                ref_tmp_fin = np.where(
                    ref_tmp_fin > -1,
                    np.ones_like(ref_tmp_fin),
                    np.zeros_like(ref_tmp_fin),
                )
                pred_prob_tmp_fin = np.asarray(df_matching["pred_prob"])
                if self.per_case:
                    if len(self.measures_pcc) > 0:
                        BPM = BinaryPairwiseMeasures(
                            pred=pred_tmp_fin,
                            ref=ref_tmp_fin,
                            measures=self.measures_pcc,
                            dict_args=self.dict_args,
                        )
                        det_res = BPM.to_dict_meas()
                        det_res["label"] = lab
                        det_res["case"] = name
                        list_det.append(det_res)
                    if len(self.measures_mt) > 0:
                        PPM = ProbabilityPairwiseMeasures(
                            pred_prob_tmp_fin,
                            ref_tmp_fin,
                            measures=self.measures_mt,
                            dict_args=self.dict_args,
                        )
                        mt_res = PPM.to_dict_meas()
                        mt_res["label"] = lab
                        mt_res["case"] = name
                        list_mt.append(mt_res)
                else:
                    list_pred.append(pred_tmp_fin)
                    list_ref.append(ref_tmp_fin)
                    list_prob.append(pred_prob_tmp_fin)
            if not self.per_case:
                overall_pred = np.concatenate(list_pred)
                overall_ref = np.concatenate(list_ref)
                overall_prob = np.concatenate(list_prob)
                if len(self.measures_pcc) > 0:
                    BPM = BinaryPairwiseMeasures(
                        pred=overall_pred,
                        ref=overall_ref,
                        measures=self.measures_pcc,
                        dict_args=self.dict_args,
                    )
                    det_res = BPM.to_dict_meas()
                    det_res["label"] = lab
                    list_det.append(det_res)
                if len(self.measures_mt) > 0 and self.flag_valid_proba:
                    PPM = ProbabilityPairwiseMeasures(
                        overall_prob,
                        ref_tmp_fin,
                        measures=self.measures_mt,
                        dict_args=self.dict_args,
                    )
                    mt_res = PPM.to_dict_meas()
                    mt_res["label"] = lab
                    list_mt.append(mt_res)
        return pd.DataFrame.from_dict(list_det), pd.DataFrame.from_dict(list_mt)


class MultiLabelPairwiseMeasures(object):
    # Semantic segmentation or Image wide classification
    def __init__(
        self,
        pred,
        ref,
        pred_proba,
        list_values,
        names=[],
        measures_pcc=[],
        measures_mt=[],
        measures_mcc=[],
        measures_overlap=[],
        measures_boundary=[],
        measures_calibration=[],
        connectivity_type=1,
        per_case=False,
        pixdim=[1, 1, 1],
        empty=False,
        dict_args={},
    ):
        self.pred = pred
        self.pred_proba = pred_proba
        self.flag_valid_proba = True
        self.ref = ref
        self.list_values = list_values
        self.measures_binary = measures_pcc + measures_overlap + measures_boundary
        self.measures_mcc = measures_mcc
        self.measures_mt = measures_mt
        self.measures_calibration = measures_calibration
        self.connectivity_type = connectivity_type
        if len(self.pred)>0:
            ndim = np.asarray(self.pred[0]).ndim
        self.pixdim = pixdim[:ndim]
        self.dict_args = dict_args
        self.per_case = per_case
        self.names = names
        if len(self.names) < len(self.ref):
            self.names = range(len(self.ref))
        
        if pred_proba is None or pred_proba[0] is None:
            self.flag_valid_proba = False
        #print(self.names, ' is names')

    def per_label_dict(self):
        list_bin = []
        list_mt = []
        for lab in self.list_values:
            print(lab, ' is treated label')
            list_pred = []
            list_ref = []
            list_prob = []
            list_case = []
            for (case, name) in zip(range(len(self.ref)), self.names):
                pred_case = np.asarray(self.pred[case])
                ref_case = np.asarray(self.ref[case])
                if self.pred_proba[case] is not None:
                    prob_case = np.asarray(self.pred_proba[case])
                else:
                    prob_case = None
                #print(pred_case, ' is case ', case)
                pred_tmp = np.where(
                    pred_case == lab, np.ones_like(pred_case), np.zeros_like(pred_case)
                )
                
                #print(prob_case)
                if prob_case is not None:
                    pred_proba_tmp = prob_case[...,lab]
                else:
                    pred_proba_tmp = None
                    self.flag_valid_proba = False
                
                ref_tmp = np.where(
                    ref_case == lab, np.ones_like(ref_case), np.zeros_like(ref_case)
                )
                #print(pred_tmp, ref_tmp)
                if self.per_case:
                    if len(self.measures_binary) > 0:
                        BPM = BinaryPairwiseMeasures(
                            pred_tmp,
                            ref_tmp,
                            measures=self.measures_binary,
                            connectivity_type=self.connectivity_type,
                            pixdim=self.pixdim,
                            dict_args=self.dict_args,
                        )
                        dict_bin = BPM.to_dict_meas()
                        dict_bin["label"] = lab
                        dict_bin["case"] = name
                        list_bin.append(dict_bin)
                    if self.flag_valid_proba and len(self.measures_mt)>0:
                        PPM = ProbabilityPairwiseMeasures(
                        pred_proba=pred_proba_tmp,
                        ref_proba=ref_tmp,
                        measures=self.measures_mt,
                        dict_args=self.dict_args,
                        )
                        dict_mt = PPM.to_dict_meas()
                        dict_mt["label"] = lab
                        dict_mt["case"] = name
                        list_mt.append(dict_mt)
                    else:
                        warnings.warn('No probabilistic input or no probabilistic measure so impossible to get multi-threshold metric')
                else:
                    list_pred.append(pred_tmp)
                    list_ref.append(ref_tmp)
                    if self.flag_valid_proba:
                        list_prob.append(pred_proba_tmp)
                    list_case.append(np.ones_like(pred_case) * case)
            if not self.per_case:
                overall_pred = np.concatenate(list_pred)
                overall_ref = np.concatenate(list_ref)
                if self.flag_valid_proba:
                    overall_prob = np.concatenate(list_prob)
                else:
                    overall_prob = None
                if len(self.measures_binary)>0:
                    BPM = BinaryPairwiseMeasures(
                        overall_pred,
                        overall_ref,
                        measures=self.measures_binary,
                        connectivity_type=self.connectivity_type,
                        pixdim=self.pixdim,
                        dict_args=self.dict_args,
                    )
                    dict_bin = BPM.to_dict_meas()
                    dict_bin["label"] = lab
                    list_bin.append(dict_bin)
                
                if len(self.measures_mt)>0 and self.flag_valid_proba:
                    PPM = ProbabilityPairwiseMeasures(
                        overall_prob,
                        overall_ref,
                        case=list_case,
                        measures=self.measures_mt,
                        dict_args=self.dict_args,
                    )
                    dict_mt = PPM.to_dict_meas()
                    dict_mt["label"] = lab
                    
                    list_mt.append(dict_mt)

        return pd.DataFrame.from_dict(list_bin), pd.DataFrame.from_dict(list_mt)

    def multi_label_res(self):
        list_pred = []
        list_ref = []
        list_prob = []
        list_mcc = []
        list_cal = []
        for (case, name) in zip(range(len(self.ref)), self.names):
            pred_case = np.asarray(self.pred[case])
            ref_case = np.asarray(self.ref[case])
            if self.pred_proba[case] is not None and self.flag_valid_proba:
                prob_case = np.asarray(self.pred_proba[case])
            else:
                prob_case = None
            if self.per_case:
                if len(self.measures_mcc) > 0:
                    MPM = MultiClassPairwiseMeasures(
                        pred_case,
                        ref_case,
                        self.list_values,
                        measures=self.measures_mcc,
                        dict_args=self.dict_args,
                    )
                
                    dict_mcc = MPM.to_dict_meas()
                    dict_mcc["case"] = name
                    list_mcc.append(dict_mcc)
                if len(self.measures_calibration) > 0 and prob_case is not None:
                    CM = CalibrationMeasures(prob_case, ref_case,measures=self.measures_calibration, dict_args=self.dict_args)
                    dict_cm = CM.to_dict_meas()
                    dict_cm['case'] = name
                    list_cal.append(dict_cm)

            else:
                list_pred.append(pred_case)
                list_ref.append(ref_case)
                list_prob.append(prob_case)
        if self.per_case:
            pd_mcc = pd.DataFrame.from_dict(list_mcc)
            pd_cal = pd.DataFrame.from_dict(list_cal)
        else:
            overall_pred = np.concatenate(list_pred)
            overall_ref = np.concatenate(list_ref)
            if self.flag_valid_proba:
                overall_prob = np.concatenate(list_prob)
            else:
                overall_prob = None
            if len(self.measures_mcc) > 0:
                MPM = MultiClassPairwiseMeasures(
                    overall_pred,
                    overall_ref,
                    self.list_values,
                    measures=self.measures_mcc,
                    dict_args=self.dict_args,
                )

                dict_mcc = MPM.to_dict_meas()
                list_mcc.append(dict_mcc)
                pd_mcc = pd.DataFrame.from_dict(list_mcc)
            else:
                pd_mcc = None
            if len(self.measures_calibration) > 0 and overall_prob is not None:
                CM = CalibrationMeasures(overall_prob, overall_ref,measures=self.measures_calibration)
                dict_cal = CM.to_dict_meas()
                list_cal.append(dict_cal)
                pd_cal = pd.DataFrame.from_dict(list_cal)
            else:
                pd_cal = None
        return pd_mcc, pd_cal
