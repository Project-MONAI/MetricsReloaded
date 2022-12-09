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
Assignment localization - :mod:`MetricsReloaded.utility.assignment_localization`
====================================================================

This module provides classes for performing the :ref:`assignment and localization  <assignloc>`
required in instance segmentation and object detection tasks .

.. _assignloc:

Performing the process associated with instance segmentation
------------------------------------

.. autoclass:: AssignmentMapping
    :members:
"""


import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist
import warnings
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
from MetricsReloaded.utility.utils import (
    intersection_boxes,
    area_box,
    union_boxes,
    box_ior,
    box_iou,
    guess_input_style,
    com_from_box,
    compute_center_of_mass,
    compute_box,
    point_in_box,
    point_in_mask,
)

__all__ = [
    "AssignmentMapping",
]


class AssignmentMapping(object):
    """
    Class allowing the assignment and localization of individual objects of interests.
    The localization strategies are either based on box characteristics:
    - box_iou
    - box_ior
    - box_com
    or on the masks
    - mask_iou
    - mask_ior
    - mask_com
    - boundary_iou
    or using only centre of mass
    - com_dist
    or a mix of mask and box
    - point_in_box
    or of point and mask
    - point_in_mask
    where iou refers to Intersection over Union, IoR to Intersection over Reference, and CoM to Centre of Mass
    Options to solve assignment ambiguities are one of the following:
    - hungarian: minimising assignment cost
    - greedy_matching: based on best matching
    - greedy_performance: based on probability score
    flag_fp_in indicates whether or not to consider the double detection of reference objects as false positives or not 
    """

    def __init__(
        self,
        pred_loc,
        ref_loc,
        pred_prob,
        localization="box_iou",
        thresh=0.5,
        assignment="greedy_matching",
        flag_fp_in=True
    ):

        self.pred_loc = np.asarray(pred_loc)
        self.pred_prob = pred_prob
        self.ref_loc = np.asarray(ref_loc)
        self.localization = localization
        self.assignment = assignment
        self.thresh = thresh
        self.flag_fp_in = flag_fp_in
        flag_usable, flag_predmod, flag_refmod = self.check_input_localization()
        # self.pred_class = pred_class
        
        # self.ref_class = ref_class
        self.flag_usable = flag_usable
        self.flag_predmod = flag_predmod
        self.flag_refmod = flag_refmod
        if self.flag_usable:
            if localization == "box_iou":
                self.matrix = self.pairwise_boxiou()
            elif localization == "box_com":
                self.matrix = self.pairwise_pointcomdist()
            elif localization == "box_ior":
                self.matrix = self.pairwise_boxior()
            elif localization == "mask_iou":
                self.matrix = self.pairwise_maskiou()
            elif localization == "mask_ior":
                self.matrix = self.pairwise_maskior()
            elif localization == "mask_com":
                self.matrix = self.pairwise_maskcom()
            elif localization == "boundary_iou":
                self.matrix = self.pairwise_boundaryiou()
            elif localization == "point_in_mask":
                self.matrix = self.pairwise_pointinmask()
            elif localization == "point_in_box":
                self.matrix = self.pairwise_pointinbox()
            elif localization == "com_dist":
                self.matrix = self.pairwise_pointcomdist()
            else:
                self.flag_usable = False
                warnings.warn("No adequate localization strategy chosen - not going ahead")
        
        if self.localization in ['point_in_mask','point_in_box']:
            if self.assignment == 'greedy_matching':
                self.flag_usable = False
                warnings.warn("The localization strategy does not provide grading. Impossible to base assignment on localization performance!")
        if self.flag_usable:        
            self.df_matching, self.valid = self.resolve_ambiguities_matching()

    
    def check_input_localization(self):
        flag_refmod = False
        flag_predmod = False
        flag_usable = True
        warnings.warn("We assume that all prediction are in the same format. We also assume that and reference location are in the same format")
        if self.ref_loc.shape[0] > 0:
            input_ref = guess_input_style(self.ref_loc[0,...])
        else:
            return flag_usable, flag_refmod, flag_predmod
        if self.pred_loc.shape[0] > 0:
            input_pred = guess_input_style(self.pred_loc[0,...])
        else:
            return flag_usable, flag_refmod, flag_predmod
        print(input_ref, input_pred)
        if self.localization == 'box_com':
            if input_ref == 'box':
                flag_refmod = True
                self.com_fromrefbox()
            if input_pred == 'box':
                flag_predmod = True
                self.com_frompredbox()
            if input_ref == 'mask':
                flag_refmod = True
                self.com_fromrefmask()
            if input_pred == 'mask':
                flag_predmod = True
                self.com_frompredmask()
        if self.localization in ['box_iou','box_ior']:
            if input_ref == 'com' or input_pred == 'com':
                warnings.warn("Input not suitable - please use localization related to com")
                flag_usable = False
                return flag_usable, flag_predmod, flag_refmod
            if input_ref == 'mask':
                warnings.warn('We will need to reprocess the reference input to be ingested as box corners')
                flag_refmod = True
                self.box_fromrefmask()
            if input_pred == 'mask':
                warnings.warn('We will need to reprocess the prediction input to be ingested as box')
                flag_predmod = True
                self.box_frompredmask()
        elif self.localization == 'point_in_mask':
            if input_ref != 'mask':
                warnings.warn('Input not suitable - ref are not masks')
                flag_usable = False
                return flag_usable, flag_predmod, flag_refmod
            if input_pred != 'com':
                warnings.warn('Input not suitable - pred not as points!')
                flag_usable = False
                return flag_usable, flag_predmod, flag_refmod
        elif self.localization == 'point_in_box':
            if input_pred != 'com':
                flag_usable = False
                warnings.warn('Input for prediction not suitable as not in a point format')
                return flag_usable, flag_predmod, flag_refmod
            if input_ref == 'com':
                flag_usable = False
                warnings.warn('Input for reference as point instead of box - not usable for this setting')
                return flag_usable, flag_predmod, flag_refmod
            if input_ref == 'mask':
                flag_refmod = True
                warnings.warn('We will need to modify ref to make it interpretable as box corners')
        elif self.localization == 'com_dist':
            if input_ref == 'mask':
                flag_refmod = True
                self.com_fromrefmask()
                warnings.warn('Mask provided for ref instead of point - will be transformed to be centre of mass')
            if input_ref == 'box': 
                self.com_fromrefbox()
                flag_refmod = True
                warnings.warn('Box provided instead of centre of mass - will modify to centre of mass for localization')
            if input_pred == 'mask':
                self.com_frompredmask()
                flag_predmod = True
                warnings.warn('Mask provided for prediction - will modify to centre of mass for localization')
            if input_pred == 'box':
                self.com_frompredbox()  
                flag_predmod = True
                warnings.warn('Box corners provided for prediction - centre of mass will be derived for localization')
        elif self.localization in ['mask_iou','mask_com','mask_ior','boundary_iou',]:
            if input_ref != 'mask' or input_pred !='mask':
                warnings.warn('Input not suitable - please use a localization strategy suitable for your input')
                flag_usable = False
        return flag_usable, flag_predmod, flag_refmod

    def com_frompredbox(self):
        list_mod = []
        for f in range(self.pred_loc.shape[0]):
            list_mod.append(com_from_box(self.pred_loc[f,...]))
        self.pred_loc_mod = np.vstack(list_mod)

    def com_fromrefbox(self):
        list_mod = []
        for f in range(self.ref_loc.shape[0]):
            list_mod.append(com_from_box(self.ref_loc[f,...]))
        self.ref_loc_mod = np.vstack(list_mod)

    def com_frompredmask(self):
        list_mod = []
        for f in range(self.pred_loc.shape[0]):
            list_mod.append(compute_center_of_mass(self.pred_loc[f,...]))
        self.pred_loc_mod = np.vstack(list_mod)

    def com_fromrefmask(self):
        list_mod = []
        for f in range(self.ref_loc.shape[0]):
            list_mod.append(compute_center_of_mass(self.ref_loc[f,...]))
        self.ref_loc_mod = np.vstack(list_mod)

    def box_fromrefmask(self):
        list_mod = []
        for f in range(self.ref_loc.shape[0]):
            list_mod.append(compute_box(self.ref_loc[f,...]))
        
        self.ref_loc_mod = np.vstack(list_mod)

    def box_frompredmask(self):
        list_mod = []
        for f in range(self.pred_loc.shape[0]):
            list_mod.append(compute_box(self.pred_loc[f,...]))
        self.pred_loc_mod = np.vstack(list_mod)

    def pairwise_pointcomdist(self):
        """
        Creates a matrix of size numb_prediction elements x number of reference elements
        indicating the pairwise distance of the centre of mass of the location boxes
        """
        pred_coms = self.pred_loc
        ref_coms = self.ref_loc
        if self.flag_refmod:
            ref_coms = self.ref_loc_mod
        if self.flag_predmod:
            pred_coms = self.pred_loc_mod
        matrix_cdist = cdist(pred_coms, ref_coms)
        return matrix_cdist

    
    def pairwise_pointinbox(self):
        """
        Creates a matrix of size number of prediction elements x number of reference elements
        indicating binarily whether the point representing the prediction element is in the reference box
        """
        ref_boxes = self.ref_loc
        pred_points = self.pred_loc
        if self.flag_refmod:
            ref_boxes = self.ref_loc_mod
        if self.flag_predmod:
            pred_points = self.pred_loc_mod
        matrix_pinb = np.zeros([pred_points.shape[0],ref_boxes.shape[0]])
        for (p, p_point) in enumerate(pred_points):
            for (r, r_box) in enumerate(ref_boxes):
                matrix_pinb[p,r] = point_in_box(p_point, r_box)
        return matrix_pinb

    def pairwise_pointinmask(self):
        ref_masks = self.ref_loc
        pred_points = self.pred_loc
        if self.flag_refmod:
            ref_masks = self.ref_loc_mod
        if self.flag_predmod:
            pred_points = self.pred_loc_mod
        matrix_pinm = np.zeros([pred_points.shape[0],ref_masks.shape[0]])
        for (p,p_point) in enumerate(pred_points):
            for (r,r_mask) in enumerate(ref_masks):
                matrix_pinm[p, r] = point_in_mask(p_point, r_mask)
        return matrix_pinm


    def pairwise_boxiou(self):
        """
        Creates a matrix of size number of prediction elements x number of reference elements
        indicating the pairwise box iou
        """
        ref_box = self.ref_loc
        pred_box = self.pred_loc
        print(self.flag_refmod, self.flag_predmod)
        if self.flag_refmod:
            ref_box = self.ref_loc_mod
        
        if self.flag_predmod:
            pred_box = self.pred_loc_mod
        print(ref_box, pred_box)
        matrix_iou = np.zeros([pred_box.shape[0], ref_box.shape[0]])
        for (pi, pb) in enumerate(pred_box):
            for (ri, rb) in enumerate(ref_box):
                matrix_iou[pi, ri] = box_iou(pb, rb)
        return matrix_iou

    def pairwise_maskior(self):
        """
        Creates a matrix of size number of prediction elements x number of reference elements
        indicating the pairwise mask ior
        """
        matrix_ior = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        for p in range(self.pred_loc.shape[0]):
            for r in range(self.ref_loc.shape[0]):
                PM = BinaryPairwiseMeasures(self.pred_loc[p, ...], self.ref_loc[r, ...])
                matrix_ior[p, r] = PM.intersection_over_reference()
        print("Matrix ior ", matrix_ior)
        return matrix_ior

    def pairwise_boundaryiou(self):
        """
        Creates a matrix of size number of prediction elements x number of reference elements
        indicating the pairwise boundary iou 
        """
        matrix_biou = np.zeros([self.pred_loc.shape[0],self.ref_loc.shape[0]])
        for p in range(self.pred_loc.shape[0]):
            for r in range(self.ref_loc.shape[0]):
                PM = BinaryPairwiseMeasures(self.pred_loc[p,...], self.ref_loc[r,...])
                matrix_biou[p,r] = PM.boundary_iou()
        print("Matrix Boundary IoU", matrix_biou)
        return matrix_biou

    def pairwise_maskcom(self):
        """
        Creates a matrix of size number of prediction elements x number of reference elements
        indicating the pairwise distance between mask centre of mass
        """
        matrix_com = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        for p in range(self.pred_loc.shape[0]):
            for r in range(self.ref_loc.shape[0]):
                PM = BinaryPairwiseMeasures(self.pred_loc[p, ...], self.ref_loc[r, ...])
                matrix_com[p, r] = PM.com_dist()
        print("Matrix com ", matrix_com)
        return matrix_com

    def pairwise_maskiou(self):
        """
        Creates a matrix of size number of prediction elements x number of reference elements
        indicating the pairwise mask iou.
        """
        matrix_iou = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        print(matrix_iou.shape, self.pred_loc.shape, self.ref_loc.shape)
        for p in range(self.pred_loc.shape[0]):
            for r in range(self.ref_loc.shape[0]):
                PM = BinaryPairwiseMeasures(self.pred_loc[p, ...], self.ref_loc[r, ...])
                matrix_iou[p, r] = PM.intersection_over_union()
        return matrix_iou

    def pairwise_boxior(self):
        """
        Creates a matrix of size number of prediction elements x number of reference elements
        indicating the pairwise box ior
        """
        ref_boxes = self.ref_loc
        pred_boxes = self.pred_loc
        if self.flag_refmod:
            ref_boxes = self.ref_loc_mod
        if self.flag_predmod:
            pred_boxes = self.pred_loc_mod
        matrix_ior = np.zeros([pred_boxes.shape[0], ref_boxes.shape[0]])
        for (pi, pb) in enumerate(pred_boxes):
            for (ri, rb) in enumerate(ref_boxes):
                matrix_ior[pi, ri] = box_ior(pb, rb)
        return matrix_ior

    def initial_mapping(self):
        """
        Identifies an original ideal mapping between references and prediction element for all those
        when there is no ambiguity in the assignment (only one to one matching available). Creates the list of
        possible options when multiple are possible and populates the relevant dataframes with performance of the
        localization metrics and the assigned score probability.
        """
        matrix = self.matrix
        if self.localization in ['mask_com', 'box_com','com_dist']:
            possible_binary = np.where(
                matrix < self.thresh, np.ones_like(matrix), np.zeros_like(matrix)
            )
        else:
            possible_binary = np.where(
                matrix > self.thresh, np.ones_like(matrix), np.zeros_like(matrix)
            )

        list_valid = []
        list_matching = []
        list_notexist = []
        print(possible_binary.shape)
        for f in range(possible_binary.shape[0]):
            ind_possible = np.where(possible_binary[f, :] == 1)
            if len(ind_possible[0]) == 0:
                new_dict = {}
                new_dict["pred"] = f
                # new_dict['pred_class'] = self.pred_class[f]
                new_dict["pred_prob"] = self.pred_prob[f]
                new_dict["ref"] = -1
                # new_dict['ref_class'] = -1
                new_dict["performance"] = np.nan
                list_notexist.append(new_dict)
            elif len(ind_possible[0]) == 1:
                new_dict = {}
                new_dict["pred"] = f
                # new_dict['pred_class'] = self.pred_class[f]
                new_dict["pred_prob"] = self.pred_prob[f]
                new_dict["ref"] = ind_possible[0][0]
                # new_dict['ref_class'] = [self.ref_class[r] for r in ind_possible[0]]
                new_dict["performance"] = matrix[f, ind_possible[0][0]]
                list_matching.append(new_dict)
                list_valid.append(f)
            else:
                for i in ind_possible[0]:
                    new_dict = {}
                    new_dict["pred"] = f
                    # new_dict['pred_class'] = self.pred_class[f]
                    new_dict["pred_prob"] = self.pred_prob[f]
                    new_dict["ref"] = i
                    # new_dict['ref_class'] = self.ref_class[i]
                    new_dict["performance"] = matrix[f, i]
                    list_matching.append(new_dict)
                list_valid.append(f)
        df_matching = pd.DataFrame.from_dict(list_matching)
        if df_matching.shape[0] > 0:
            list_ref = list(df_matching["ref"])
        else:
            list_ref = []
        missing_ref = [r for r in range(len(self.ref_loc)) if r not in list_ref]
        list_missing = []
        for f in missing_ref:
            new_dict = {}
            new_dict["pred"] = -1
            new_dict["pred_prob"] = 0
            # new_dict['pred_class'] = -1
            new_dict["ref"] = f
            # new_dict['ref_class'] = self.ref_class[f]
            new_dict["performance"] = np.nan
            list_missing.append(new_dict)
        df_fn = pd.DataFrame.from_dict(list_missing)
        df_fp = pd.DataFrame.from_dict(list_notexist)
        # df_matching_all = pd.concat(df_matching, df_missing)
        return df_matching, df_fn, df_fp, list_valid

    def resolve_ambiguities_matching(self):
        """
        Finalise the mapping based on the initial guess by deciding on the possible ambiguities
        Returns a final pandas dataframe with all attribution and erroneous detection / non detections.

        """
        matrix = self.matrix
        df_matching, df_fn, df_fp, list_valid = self.initial_mapping()
        print(
            "Number of categories: TP FN FP",
            df_matching.shape,
            df_fn.shape,
            df_fp.shape,
        )
        df_ambiguous_ref = None
        if df_matching.shape[0] > 0:
            df_matching["count_pred"] = df_matching.groupby("ref")["pred"].transform(
                "count"
            )
            df_matching["count_ref"] = df_matching.groupby("pred")["ref"].transform(
                "count"
            )
            df_ambiguous_ref = df_matching[
                (df_matching["count_ref"] > 1) & (df_matching["ref"] > -1)
            ]
            df_ambiguous_seg = df_matching[
                (df_matching["count_pred"] > 1) & (df_matching["pred"] > -1)
            ]
        if (
            df_ambiguous_ref is None
            or df_ambiguous_ref.shape[0] == 0
            and df_ambiguous_seg.shape[0] == 0
        ):
            print("No ambiguity in matching")
            df_matching_all = pd.concat([df_matching, df_fp, df_fn])
            return df_matching_all, list_valid
        else:
            if self.assignment == "hungarian":
                valid_matrix = matrix[list_valid, :]
                if self.localization not in ['mask_com', 'box_com','com_dist'] :
                    valid_matrix = 1 - valid_matrix
                row, col = lsa(valid_matrix)
                list_matching = []
                for (r, c) in zip(row, col):
                    df_tmp = df_matching[
                        df_matching["seg"] == list_valid[r] & (df_matching["ref"] == c)
                    ]
                    list_matching.append(df_tmp)
                df_ordered2 = pd.concat(list_matching)
            elif self.assignment == "greedy_matching":
                if self.localization not in ['mask_com','box_com','com_dist'] :
                    df_ordered = df_matching.sort_values("performance").drop_duplicates(
                        "pred"
                    )
                    df_ordered2 = df_ordered.sort_values("performance").drop_duplicates(
                        ["ref"]
                    )
                else:
                    df_ordered = df_matching.sort_values(
                        "performance", ascending=False
                    ).drop_duplicates("pred")
                    df_ordered2 = df_ordered.sort_values(
                        "performance", ascending=False
                    ).drop_duplicates("ref")
            else:
                df_ordered = df_matching.sort_values(
                    "pred_prob", ascending=False
                ).drop_duplicates("pred")
                df_ordered2 = df_ordered.sort_values(
                    "pred_prob", ascending=False
                ).drop_duplicates("ref")
            list_seg_not = [
                s
                for s in list(df_matching["pred"])
                if s not in list(df_ordered2["pred"])
            ]
            list_ref_not = [
                r for r in range(matrix.shape[1]) if r not in list(df_ordered2["ref"])
            ]
            list_pred_fp = []
            list_ref_fn = []
            for f in list_seg_not:
                new_dict = {}
                new_dict["pred"] = f
                new_dict["pred_prob"] = self.pred_prob[f]
                new_dict["ref"] = -1
                new_dict["performance"] = np.nan
                list_pred_fp.append(new_dict)
            for r in list_ref_not:
                new_dict = {}
                new_dict["pred"] = -1
                new_dict["pred_prob"] = 0
                new_dict["ref"] = r
                new_dict["performance"] = np.nan
                list_ref_fn.append(new_dict)
            df_fp_new = pd.DataFrame.from_dict(list_pred_fp)
            df_fn_all = pd.DataFrame.from_dict(list_ref_fn)
            if self.flag_fp_in:
                df_matching_all = pd.concat([df_ordered2, df_fn_all, df_fp, df_fp_new])
            else:
                df_matching_all = pd.concat([df_ordered2, df_fp, df_fn_all])
            return df_matching_all, list_valid

    def matching_ref_predseg(self):
        """
        In case mask of individual elements are available (Instance segmentation task)
        provides the list of true positive prediction, associated list of reference segmentation,
        list of false positive masks and of false negative masks as
        returns: list_pred, list_ref, list_fp, list_fn
        """
        df_matching_all = self.df_matching
        df_tp = df_matching_all[
            (df_matching_all["ref"] >= 0) & (df_matching_all["pred"] >= 0)
        ]
        df_fp = df_matching_all[(df_matching_all["ref"] < 0)]
        df_fn = df_matching_all[(df_matching_all["pred"] < 0)]
        list_pred = []
        list_ref = []
        list_fp = []
        list_fn = []
        for r in range(df_tp.shape[0]):
            print(df_tp.iloc[r]["pred"])
            list_pred.append(self.pred_loc[int(df_tp.iloc[r]["pred"]), ...])
            list_ref.append(self.ref_loc[int(df_tp.iloc[r]["ref"]), ...])
        for r in range(df_fp.shape[0]):
            list_fp.append(self.pred_loc[int(df_fp.iloc[r]["pred"]), ...])
        for r in range(df_fn.shape[0]):
            list_fn.append(self.ref_loc[int(df_fn.iloc[r]["ref"]), ...])
        return list_pred, list_ref, list_fp, list_fn
