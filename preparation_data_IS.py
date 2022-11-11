import glob
from MetricsReloaded.processes.overall_process import ProcessEvaluation
import os
import nibabel as nib
import pickle as pkl
import pandas as pd
from MetricsReloaded.utility.utils import MorphologyOps


#list_reffile = glob.glob('examples/data/Ref/CorrectLesion*CLA*')
#list_predfile = glob.glob('examples/data/Pred/CorrectLesion*RAP94*')
list_reffile = glob.glob('examples/data/Ref/CorrectLesion*')
list_predfile = glob.glob('examples/data/Pred/CorrectLesion*')
list_det = []
list_seg = []
for f in list_reffile:
    name = os.path.split(f)[1]
    name = name.split('Ref')[0]
    name = name.split('Lesion_')[1]
    if not os.path.exists('examples/results/Det94AvFin_%s.csv'%name):
    
        list_pospred = [c for c in list_predfile if name in c]
        if len(list_pospred) == 1:
            ref = nib.load(f).get_fdata()
            pred = nib.load(list_pospred[0]).get_fdata()
            #ref = nib.load('/Users/csudre/Data/B-RAPIDD/CLA66/CorrectLesion_B-RAP_0007_01_CLA66.nii.gz').get_fdata()
            #pred = nib.load('/Users/csudre/Data/B-RAPIDD/RAP66/CorrectLesion_B-RAP_0007_01_RAP66.nii.gz').get_fdata()
            ref_bin = ref>=0.5
            pred_bin = pred>=0.5
            print(f,list_pospred[0])
            # Splitting the mask into it's component instences
            list_ref,_,_ = MorphologyOps(ref_bin, neigh=6).list_foreground_component()
            list_pred,_,_ = MorphologyOps(pred_bin,neigh=6).list_foreground_component()
            pred_class = [1]*len(list_pred)
            pred_prob = [1]*len(list_pred)
            ref_class = [1]*len(list_ref)
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
            # This does the evaluation and saves the TP FP FN (T= True, P = positive,.. )
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
                measures_boundary=['masd','nsd','boundary_iou'],
                measures_detseg=['PQ'],
                thresh_ass=0.000001
            )
            df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()
            df_resdet['id'] = name
            # This line gives an error: Cannot save file into a non-existent directory: 'examples\results'
            df_resseg['id'] = name
            df_resdet.to_csv('examples/results/Det94AvFin_%s.csv'%name)
            df_resseg.to_csv('examples/results/Seg94AvFin_%s.csv' %name)
            
            list_det.append(df_resdet)
            list_seg.append(df_resseg)
df_resdetall = pd.concat(list_det)
df_ressegall = pd.concat(list_seg)
df_resdetall.to_csv('examples/results/Det94AvFin.csv')
df_ressegall.to_csv('examples/results/Seg94AvFin.csv')
print(df_resdet, df_resseg)




