import glob
from processes.overall_process import ProcessEvaluation
import os
import nibabel as nib
import pickle as pkl
import pandas as pd
from metrics.pairwise_measures import MorphologyOps


list_reffile = glob.glob('/Users/csudre/Data/B-RAPIDD/CLA94/CorrectLesion*')
list_predfile = glob.glob('/Users/csudre/Data/B-RAPIDD/RAP94/CorrectLesion*')
list_det = []
list_seg = []
for f in list_reffile:
    name = os.path.split(f)[1]
    name = name.split('CLA94')[0]
    name = name.split('Lesion_')[1]
    if not os.path.exists('/Users/csudre/Data/B-RAPIDD/Det94_%s.csv'%name):
    
        list_pospred = [c for c in list_predfile if name in c]
        if len(list_pospred) == 1:
            ref = nib.load(f).get_fdata()
            pred = nib.load(list_pospred[0]).get_fdata()
            #ref = nib.load('/Users/csudre/Data/B-RAPIDD/CLA66/CorrectLesion_B-RAP_0007_01_CLA66.nii.gz').get_fdata()
            #pred = nib.load('/Users/csudre/Data/B-RAPIDD/RAP66/CorrectLesion_B-RAP_0007_01_RAP66.nii.gz').get_fdata()
            ref_bin = ref>=0.5
            pred_bin = pred>=0.5
            print(f,list_pospred[0])
            list_ref = MorphologyOps(ref_bin, neigh=6).list_foreground_component()
            list_pred = MorphologyOps(pred_bin,neigh=6).list_foreground_component()
            pred_prob = []
            pred_class = []
            ref_class = []
            for k in list_pred:
                pred_class.append(1)
                pred_prob.append(1)
            for k in list_ref:
                ref_class.append(1)
            
            list_values = [1]
            file=list_pospred
            dict_file = {}
            dict_file['pred_loc'] = [list_pred]
            dict_file['ref_loc'] = [list_ref]
            dict_file['pred_prob'] = [pred_prob]
            dict_file['ref_class'] = [ref_class]
            dict_file['pred_class'] = [pred_class]
            dict_file['list_values'] = list_values
            dict_file['file'] = file
            #f = open("TestDataBRAP_%s.pkl"%name, "wb")  # Pickle file is newly created where foo1.py is
            #pkl.dump(dict_file, f)  # dump data to f
            #f.close()
            PE = ProcessEvaluation(
                dict_file,
                "Instance Segmentation",
                localization="mask_iou",
                file=list_pospred,
                flag_map=True,
                assignment="greedy_matching",
                measures_overlap=['fbeta','numb_ref','numb_pred','numb_tp','numb_fp','numb_fn'],
                measures_mcc=[],
                measures_pcc=["fbeta",'numb_ref','numb_pred','numb_tp','numb_fp','numb_fn'],
                measures_mt=[],
                measures_boundary=['masd','nsd'],
                thresh_ass=0.000001
            )
            df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()
            df_resdet['id'] = name
            df_resseg['id'] = name
            df_resdet.to_csv('/Users/csudre/Data/B-RAPIDD/Det94_%s.csv'%name)
            df_resseg.to_csv('/Users/csudre/Data/B-RAPIDD/Seg94_%s.csv' %name)
            
            list_det.append(df_resdet)
            list_seg.append(df_resseg)
df_resdetall = pd.concat(list_det)
df_ressegall = pd.concat(list_seg)
df_resdetall.to_csv('/Users/csudre/Data/B-RAPIDD/Det94.csv')
df_ressegall.to_csv('/Users/csudre/Data/B-RAPIDD/Seg94.csv')
print(df_resdet, df_resseg)




seg_box = [[1, 1, 4, 4], [5, 6, 7, 6], [3, 5, 4, 6]]
ref_box = [[1, 1, 3, 3], [6, 6, 6, 6], [3, 5, 4, 5], [4, 5, 6, 6]]
seg_prob = [0.4, 0.8, 0.9]
seg_class = [1, 1, 1]
ref_class = [1, 1, 2, 1]
dict_file = {}
dict_file["pred_loc"] = [seg_box]
dict_file["ref_loc"] = [ref_box]
dict_file["pred_prob"] = [seg_prob]
dict_file["pred_class"] = [seg_class]
dict_file["ref_class"] = [ref_class]
dict_file["list_values"] = [1, 2]
f = open("TestDataMR.pkl", "wb")  # Pickle file is newly created where foo1.py is
pkl.dump(dict_file, f)  # dump data to f
f.close()
PE = ProcessEvaluation(
    "TestDataMR.pkl",
    "Object Detection",
    localization="box_iou",
    assignment="greedy_matching",
    measures_overlap=[],
    measures_mcc=[],
    measures_pcc=["fbeta"],
    measures_mt=["ap"],
    measures_boundary=[],
)
df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()
print(df_resdet, df_resseg, df_resmt)

pred_ilc = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
ref_ilc = [0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 2, 2]
pred_prob = [
    0.4,
    0.5,
    0.5,
    0.75,
    0.9,
    0.8,
    0.5,
    0.4,
    0.9,
    0.8,
    0.8,
    0.7,
    0.6,
    0.5,
    0.6,
    0.7,
    0.9,
    1,
]
dict_file = {}
dict_file["pred_class"] = [pred_ilc]
dict_file["ref_class"] = [ref_ilc]
dict_file["pred_prob"] = [pred_prob]
dict_file["list_values"] = [0, 1, 2]
f = open("TestILC.pkl", "wb")  # Pickle file is newly created where foo1.py is
pkl.dump(dict_file, f)  # dump data to f
f.close()

PE = ProcessEvaluation(
    "TestILC.pkl",
    "Image Classification",
    localization=None,
    assignment=None,
    measures_pcc=["lr+"],
    measures_mt=["auroc", "sens@spec"],
    measures_mcc=["mcc"],
)
df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()