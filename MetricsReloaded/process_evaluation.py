# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from MetricsReloaded.utility.assignment_localization import AssignmentMapping
from MetricsReloaded.processes.mixed_measures_processes import (
    MultiLabelLocMeasures,
    MultiLabelLocSegPairwiseMeasure,
    MultiLabelPairwiseMeasures,
)
from MetricsReloaded.metrics.pairwise_measures import (
    BinaryPairwiseMeasures,
    MorphologyOps,
)
from MetricsReloaded.utility.utils import combine_df

# from assignment_localization import AssignmentMapping
import numpy as np
from numpy.random import default_rng
from sklearn.metrics import roc_auc_score
import pickle as pkl
import nibabel as nib
import glob
import os
import pandas as pd
import argparse
import sys
import warnings

dict_task_mapping = {
    "IS": "Instance Segmentation",
    "SS": "Semantic Segmentation",
    "OD": "Object Detection",
    "ILC": "Image Classification",
}

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


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press ⌘F8 to toggle the breakpoint.


def parse_options(string_options):
    dict_args = {}
    for f in string_options:
        key = f.split(":")[0]
        value = f.split(":")[1]
        dict_args[key] = np.float(value)
    return dict_args


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
        measures_calibration=[],
        flag_map=False,
        file=[],
        thresh_ass=0.5,
        case=False,
        flag_fp_in=True,
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
        self.measures_calibration = measures_calibration
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
                measures_calibration=self.measures_calibration,
                list_values=data["list_values"],
                per_case=self.case,
            )
            df_bin, df_mt, df_cal = MLPM.per_label_dict()
            df_mcc = MLPM.multi_label_res()
            if self.category == "Image Classification":
                df_resdet = df_bin
                df_resseg = None
                df_resmt = df_mt
                df_resmcc = df_mcc
                df_rescal= df_cal
            else:
                df_resdet = None
                df_resseg = df_bin
                df_resmt = df_mt
                df_resmcc = df_mcc
                df_rescal = None
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




def main(argv):
    print(argv)
    parser = argparse.ArgumentParser(description="Process evaluation")

    parser.add_argument(
        "-i",
        dest="input",
        metavar="input file",
        type=str,
        default="",
        help="Input pickle file where the data to use for evaluation is stored",
    )
    parser.add_argument(
        "-o",
        dest="output",
        metavar="output pattern",
        type=str,
        default="",
        help="RegExp pattern for the hd5 output files (output of the merging script mergin_script.py",
    )
    parser.add_argument(
        "-task",
        dest="task",
        metavar="csv fwith selected list of ids",
        type=str,
        choices=["ILC", "IS", "SS", "OD"],
        default="ILC",
        help="Task for the evaluation: ILC for Image Level classification, IS for Instance Segmentation, SS for semantic segmentation and OD for object detection",
    )

    parser.add_argument(
        "-localization",
        dest="localization",
        action="store",
        choices=["mask_iou", "mask_ior", "box_iou", "box_ior", "box_com", "mask_com"],
        default="box_iou",
        type=str,
        help="Choice of localisation method",
    )

    parser.add_argument(
        "-thresh_loc",
        dest="thresh_loc",
        action="store",
        default=0,
        type=float,
        help="Choice of threshold for the localization strategy default is 0",
    )

    parser.add_argument(
        "-assignment",
        dest="assignment",
        action="store",
        choices=["greedy_matching", "greedy_score", "hungarian"],
        default="greedy_matching",
        type=str,
        help="result path",
    )

    parser.add_argument(
        "-mt",
        dest="mt",
        type=str,
        nargs="+",
        action="store",
        choices=[
            "ap",
            "auroc",
            "froc",
            "sens@spec",
            "sens@ppv",
            "spec@sens",
            "fppi@sens",
            "ppv@sens",
            "sens@fppi",
        ],
        help="multi threshold measures",
        default=[],
    )

    parser.add_argument(
        "-pcc",
        dest="pcc",
        type=str,
        nargs="+",
        action="store",
        choices=["fbeta", "accuracy", "balanced_accuracy", "lr+", "youden_ind", "mcc"],
        help="per class counting metrics",
        default=[],
    )

    parser.add_argument(
        "-mcc",
        dest="mcc",
        type=str,
        nargs="+",
        action="store",
        choices=["wck", "mcc", "balanced_accuracy", "cohens_kappa"],
        help="multi class counting metrics",
        default=[],
    )

    parser.add_argument(
        "-overlap",
        dest="overlap",
        type=str,
        nargs="+",
        action="store",
        choices=["iou", "fbeta", "dsc", "centreline_dsc"],
        help="overlap metrics",
        default=[],
    )

    parser.add_argument(
        "-boundary",
        dest="boundary",
        type=str,
        nargs="+",
        choices=["masd", "assd", "hd_perc", "hd", "boundary_iou", "nsd"],
        action="store",
        help="boundary metrics among masd, assd, hd_perc, hd, boundary_iou and nsd. Options are set up using the opt_boundary field otherwise, default for hd_perc is 95 and for nsd 1",
        default=[],
    )

    parser.add_argument(
        "-opt_boundary",
        dest="opt_boundary",
        type=str,
        nargs="+",
        action="store",
        default=[],
        help="options for hd_perc and or nsd as hd_perc:perc_value and nsd:tau_value",
    )

    parser.add_argument(
        "-opt_mcc",
        dest="opt_mcc",
        type=str,
        nargs="+",
        action="store",
        default=[],
        help="option for weights of cohens kappa as wck:array_weights",
    )

    parser.add_argument(
        "-opt_pcc",
        dest="opt_pcc",
        type=str,
        nargs="+",
        action="store",
        default=[],
        help="option for fbeta as fbeta:beta_value ",
    )

    parser.add_argument(
        "-opt_mt",
        dest="opt_mt",
        type=str,
        nargs="+",
        action="store",
        default=[],
        help="option for sens@spec, sens@ppv, spec@sens, fppi@sens, ppv@sens, sens@fppi",
    )

    parser.add_argument(
        "-opt_overlap",
        dest="opt_overlap",
        type=str,
        nargs="+",
        action="store",
        default=[],
    )

    try:
        args = parser.parse_args(argv)

    except argparse.ArgumentTypeError:
        print(
            "compute_ROI_statistics.py -i <input_image_pattern> -m "
            "<mask_image_pattern> -t <threshold> -mul <analysis_type> "
            "-trans <offset>   "
        )
        sys.exit(2)

    print("Performing checks on the chosen arguments")
    if args.task != "ILC":
        if len(args.mcc) > 0:
            warnings.warn(
                "Multi Class Counting metric should only be used in an ILC task"
            )
        if len(args.mt) > 0:
            if "auroc" in args.mt:
                warnings.warn("AUROC should only be chosen for an ILC task")
            if "sens@spec" in args.mt:
                warnings.warn(
                    "Sensitivity@Specificity should only be chosen for an ILC task"
                )
            if "spec@sens" in args.mt:
                warnings.warn(
                    "Specificity@Sensitivity should only be chosen for an ILC task"
                )
        if len(args.pcc) > 0:
            if "lr+" in args.pcc:
                warnings.warn("LR+ should only be chosen for an ILC task")
    if args.task != "OD" and args.task != "IS":
        if len(args.mt) > 0:
            if "fppi@sens" in args.mt:
                warnings.warn(
                    "FPPI@Sensitivity should only be chosen for an OD or IS task"
                )
            if "sens@fppi" in args.mt:
                warnings.warn(
                    "Sensitivity@FPPI should only be chosen for an OD or IS task"
                )
            if "froc" in args.mt:
                warnings.warn("FROC score should only be chosen for an OD or IS task")
    if args.task != "IS" and args.task != "SS":
        if len(args.boundary) > 0:
            warnings.warn("Boundary metrics should only be used for an IS or SS task")
        if len(args.overlap) > 0:
            warnings.warn("Overlap metrics should only be used for an IS or SS task")
    string_options = (
        args.opt_boundary + args.opt_pcc + args.opt_overlap + args.opt_mt + args.opt_mcc
    )
    dict_args = parse_options(string_options)
    print(dict_args)
    all_measures = args.mt + args.mcc + args.overlap + args.pcc + args.boundary
    for k in dict_args.keys():
        if k not in all_measures:
            warnings.warn(
                "Option set for %s that is not in list of metrics to be calculated" % k
            )

    print("Performing checks on the provided input")
    f = open(args.input, "rb")
    data = pkl.load(f)
    f.close()
    if args.task == "ILC":
        measures_pcc = args.pcc + [
            "numb_ref",
            "numb_pred",
            "numb_tp",
            "numb_fn",
            "numb_fp",
        ]

        PE = ProcessEvaluation(
            data,
            "Image Classification",
            measures_mcc=args.mcc,
            measures_pcc=measures_pcc,
            measures_mt=args.mt,
            dict_args=dict_args,
        )
    elif args.task == "SS":
        measures_overlap = args.overlap + [
            "numb_ref",
            "numb_pred",
            "numb_tp",
            "numb_fn",
            "numb_fp",
        ]
        PE = ProcessEvaluation(
            data,
            "Semantic Segmentation",
            measures_boundary=args.boundary,
            measures_overlap=measures_overlap,
        )
    elif args.task == "IS":
        measures_overlap = args.overlap + [
            "numb_ref",
            "numb_pred",
            "numb_tp",
            "numb_fn",
            "numb_fp",
        ]
        measures_pcc = args.pcc + [
            "numb_ref",
            "numb_pred",
            "numb_tp",
            "numb_fn",
            "numb_fp",
        ]
        PE = ProcessEvaluation(
            data,
            "Instance Segmentation",
            localization=args.localization,
            assignment=args.assignment,
            measures_boundary=args.boundary,
            measures_overlap=args.overlap,
            measures_mcc=args.mcc,
            measures_pcc=measures_pcc,
            measures_mt=args.mt,
            dict_args=dict_args,
        )
    elif args.task == "OD":
        measures_pcc = args.pcc + [
            "numb_ref",
            "numb_pred",
            "numb_tp",
            "numb_fn",
            "numb_fp",
        ]
        PE = ProcessEvaluation(
            data,
            "Object Detection",
            localization=args.localization,
            assignment=args.assignment,
            measures_pcc=measures_pcc,
            measures_mt=args.mt,
            dict_args=dict_args,
        )
    df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[1:])
    print_hi("PyCharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
