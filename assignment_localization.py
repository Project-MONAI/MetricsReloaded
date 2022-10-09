import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist

from pairwise_measures import BinaryPairwiseMeasures


def intersection_boxes(box1, box2):
    """
    Intersection between two boxes given the corners
    """
    min_values = np.minimum(box1, box2)
    max_values = np.maximum(box1, box2)
    box_inter = max_values[: min_values.shape[0] // 2]
    box_inter2 = min_values[max_values.shape[0] // 2 :]
    box_intersect = np.concatenate([box_inter, box_inter2])
    box_intersect_area = np.prod(
        np.maximum(box_inter2 + 1 - box_inter, np.zeros_like(box_inter))
    )
    return np.max([0, box_intersect_area])


def area_box(box1):
    """Determines the area / volume given the coordinates of extreme corners"""
    box_corner1 = box1[: box1.shape[0] // 2]
    box_corner2 = box1[box1.shape[0] // 2 :]
    return np.prod(box_corner2 + 1 - box_corner1)


def union_boxes(box1, box2):
    """Calculates the union of two boxes given their corner coordinates"""
    value = area_box(box1) + area_box(box2) - intersection_boxes(box1, box2)
    return value


def box_iou(box1, box2):
    """Calculates the iou of two boxes given their extreme corners coordinates"""
    numerator = intersection_boxes(box1, box2)
    denominator = union_boxes(box1, box2)
    return numerator / denominator


def box_ior(box1, box2):
    numerator = intersection_boxes(box1, box2)
    denominator = area_box(box2)
    return numerator / denominator


class AssignmentMapping(object):
    def __init__(
        self,
        pred_loc,
        ref_loc,
        pred_prob,
        localization="iou",
        thresh=0.5,
        assignment="Greedy matching",
    ):
        self.pred_loc = np.asarray(pred_loc)
        self.pred_prob = pred_prob
        # self.pred_class = pred_class
        self.ref_loc = np.asarray(ref_loc)
        # self.ref_class = ref_class
        self.localization = localization
        self.assignment = assignment
        self.thresh = thresh
        if localization == "iou":
            self.matrix = self.pairwise_iou()
        elif localization == "com":
            self.matrix = self.pairwise_comdist()
        elif localization == "ior":
            self.matrix = self.pairwise_ior()
        elif localization == "maskiou":
            self.matrix = self.pairwise_maskiou()
        elif localization == "maskior":
            self.matrix = self.pairwise_maskior()
        elif localization == "maskcom":
            self.matrix = self.pairwise_maskcom()
        self.df_matching, self.valid = self.resolve_ambiguities_matching()

    def get_dimension(self):
        return np.floor(self.pred_loc.shape[0] / 2.0)

    def pairwise_comdist(self):
        matrix_cdist = cdist(self.pred_loc, self.ref_loc)
        return matrix_cdist

    def pairwise_iou(self):
        matrix_iou = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        for (pi, pb) in enumerate(self.pred_loc):
            for (ri, rb) in enumerate(self.ref_loc):
                matrix_iou[pi, ri] = box_iou(pb, rb)
        return matrix_iou

    def pairwise_maskior(self):
        matrix_ior = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        for p in range(self.pred_loc.shape[0]):
            for r in range(self.ref_loc.shape[0]):
                PM = BinaryPairwiseMeasures(self.pred_loc[p, ...], self.ref_loc[r, ...])
                matrix_ior[p, r] = PM.intersection_over_reference()
        print("Matrix ior ", matrix_ior)
        return matrix_ior

    def pairwise_maskcom(self):
        matrix_com = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        for p in range(self.pred_loc.shape[0]):
            for r in range(self.ref_loc.shape[0]):
                PM = BinaryPairwiseMeasures(self.pred_loc[p, ...], self.ref_loc[r, ...])
                matrix_com[p, r] = PM.com_dist()
        print("Matrix com ", matrix_com)
        return matrix_com

    def pairwise_maskiou(self):
        matrix_iou = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        print(matrix_iou.shape, self.pred_loc.shape, self.ref_loc.shape)
        for p in range(self.pred_loc.shape[0]):
            for r in range(self.ref_loc.shape[0]):
                PM = BinaryPairwiseMeasures(self.pred_loc[p, ...], self.ref_loc[r, ...])
                matrix_iou[p, r] = PM.intersection_over_union()
        return matrix_iou

    def pairwise_ior(self):
        matrix_ior = np.zeros([self.pred_loc.shape[0], self.ref_loc.shape[0]])
        for (pi, pb) in enumerate(self.pred_loc):
            for (ri, rb) in enumerate(self.ref_loc):
                matrix_ior[pi, ri] = box_ior(pb, rb)
        return matrix_ior

    def initial_mapping(self):
        matrix = self.matrix
        if "com" in self.localization:
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
                if self.localization != "com":
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
                if self.localization == "com":
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
            df_matching_all = pd.concat([df_ordered2, df_fn_all, df_fp, df_fp_new])
            return df_matching_all, list_valid

    def matching_ref_predseg(self):
        df_matching_all = self.df_matching
        df_tp = df_matching_all[
            (df_matching_all["ref"] >= 0) & (df_matching_all["pred"] >= 0)
        ]
        df_fp = df_matching_all[(df_matching_all['ref']<0)]
        df_fn = df_matching_all[(df_matching_all['pred']<0)]
        list_pred = []
        list_ref = []
        list_fp = []
        list_fn = []
        for r in range(df_tp.shape[0]):
            print(df_tp.iloc[r]["pred"])
            list_pred.append(self.pred_loc[int(df_tp.iloc[r]["pred"]), ...])
            list_ref.append(self.ref_loc[int(df_tp.iloc[r]["ref"]), ...])
        for r in range(df_fp.shape[0]):
            list_fp.append(self.pred_loc[int(df_fp.iloc[r]['pred']),...])
        for r in range(df_fn.shape[0]):
            list_fn.append(self.ref_loc[int(df_fn.iloc[r]['ref']),...])
        return list_pred, list_ref, list_fp, list_fn
