import glob
from process_evaluation import ProcessEvaluation
import os
import nibabel as nib
import pickle as pkl
import pandas as pd
from pairwise_measures import MorphologyOps


list_reffile = glob.glob('/Users/csudre/Data/B-RAPIDD/CLAAv/CorrectLesion*CLA*')
list_predfile = glob.glob('/Users/csudre/Data/B-RAPIDD/RAP94Av/CorrectLesion*RAP94*')
list_det = []
list_seg = []
list_pred = []
list_ref = []
pred_class = []
pred_prob = []
ref_class = []
list_files = []
for f in list_reffile:
    name = os.path.split(f)[1]
    name = name.split('CLA')[0]
    name = name.split('Lesion_')[1]
    if not os.path.exists('/Users/csudre/Data/B-RAPIDD/Det94AvSS_%s.csv'%name):
    
        list_pospred = [c for c in list_predfile if name in c]
        
        if len(list_pospred) == 1:
            ref = nib.load(f).get_fdata()
            pred = nib.load(list_pospred[0]).get_fdata()
            #ref = nib.load('/Users/csudre/Data/B-RAPIDD/CLA66/CorrectLesion_B-RAP_0007_01_CLA66.nii.gz').get_fdata()
            #pred = nib.load('/Users/csudre/Data/B-RAPIDD/RAP66/CorrectLesion_B-RAP_0007_01_RAP66.nii.gz').get_fdata()
            ref_bin = ref>=0.5
            pred_bin = pred>=0.5
            list_ref.append(ref_bin)
            list_pred.append(pred_bin)
            print(f,list_pospred[0])
            list_files.append(list_pospred[0])
            
for k in list_pred:
    pred_class.append(1)
    pred_prob.append(1)
for k in list_ref:
    ref_class.append(1)

list_values = [1]
#file=list_pospred
dict_file = {}
dict_file['pred_loc'] = list_pred
dict_file['ref_loc'] = list_ref
dict_file['pred_prob'] = list_pred
dict_file['ref_class'] = list_ref
dict_file['pred_class'] = list_pred
dict_file['list_values'] = list_values
dict_file['file'] = list_files
#f = open("TestDataBRAP_%s.pkl"%name, "wb")  # Pickle file is newly created where foo1.py is
#pkl.dump(dict_file, f)  # dump data to f
#f.close()
PE = ProcessEvaluation(
    dict_file,
    "Semantic Segmentation",
    localization="maskiou",
    file=list_files,
    flag_map=True,
    assignment="greedy_matching",
    measures_overlap=['fbeta','numb_ref','numb_pred','numb_tp','numb_fp','numb_fn'],
    measures_mcc=[],
    measures_pcc=["fbeta",'numb_ref','numb_pred','numb_tp','numb_fp','numb_fn'],
    measures_mt=[],
    case=True,
    measures_boundary=['masd','nsd','boundary_iou'],
    thresh_ass=0.000001
)
df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()
#df_resdet['id'] = name
#df_resseg['id'] = name
#df_resdet.to_csv('/Users/csudre/Data/B-RAPIDD/Det94SSAv_%s.csv'%name)
df_resseg.to_csv('/Users/csudre/Data/B-RAPIDD/Seg94SSAvFin_%s.csv' %name)

