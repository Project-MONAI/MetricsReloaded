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
from MetricsReloaded.processes.overall_process import ProcessEvaluation

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
