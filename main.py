# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from association_localization import AssociationMapping
from mixed_measures_processes import MultiLabelLocMeasures, MultiLabelLocSegPairwiseMeasure, MultiLabelPairwiseMeasures
from pairwise_measures import  BinaryPairwiseMeasures
#from association_localization import AssociationMapping
import numpy as np
from numpy.random import default_rng
from sklearn.metrics import roc_auc_score
import pickle as pkl

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

class ProcessEvaluation(object):
    def __init__(self, file_name, category, localization, assignment, measures_pcc=[],measures_mcc=[], measures_boundary=[], measures_overlap=[],measures_mt=[]):
        self.file = file_name
        self.category = category
        self.association = assignment
        self.localization = localization
        self.measures_overlap = measures_overlap
        self.measures_boundary = measures_boundary
        self.measures_mt = measures_mt
        self.measures_mcc = measures_mcc
        self.measures_pcc = measures_pcc
        self.measures_mt = measures_mt

    def process_data(self):
        f = open(self.file, 'rb')
        data = pkl.load(f)
        f.close()
        df_resdet = None
        df_resseg = None
        df_resmt = None
        df_resmcc = None
        if self.category == 'Instance Segmentation':
            MLLS =MultiLabelLocSegPairwiseMeasure(pred_loc=data['pred_loc'], ref_loc=data['ref_loc'],
                                                  pred_prob=data['pred_prob'], ref_class=data['ref_class'],
                                                  pred_class=data['pred_class'],association=self.association,measures_mt=[],
                                                  measures_pcc=[],measures_overlap=[],measures_boundary=[],list_values=data['list_values'])
            df_resseg, df_resdet = MLLS.per_label_dict()
        elif self.category == 'Object Detection':
            measures = self.measures_pcc + self.measures_mt
            MLDT = MultiLabelLocMeasures(pred_loc=data['pred_loc'], ref_loc=data['ref_loc'],
                                         pred_prob=data['pred_prob'], ref_class=data['ref_class'],
                                         pred_class=data['pred_class'],list_values=data['list_values'],
                                         measures_pcc=[],measures_mt=[])
            df_resdet, df_resmt = MLDT.per_label_dict()
            df_resseg = None
        elif self.category in ['Image Classification','Semantic Segmentation']:
            measures = self.measures_overlap  + self.measures_boundary + self.measures_pcc + self.measures_mcc
            MLPM = MultiLabelPairwiseMeasures(data['pred_class'],data['ref_class'],data['pred_prob'],measures_pcc=self.measures_pcc, measures_overlap=self.measures_overlap,measures_boundary=self.measures_boundary, measures_mcc=self.measures_mcc, measures_mt=self.measures_mt,list_values=data['list_values'])
            df_bin, df_mt = MLPM.per_label_dict()
            df_mcc = MLPM.multi_label_res()
            if self.category == 'Image Classification':
                df_resdet = df_bin
                df_resseg = None
                df_resmt=df_mt
                df_resmcc = df_mcc
            else:
                df_resdet = None
                df_resseg = df_bin
                df_resmt = df_mt
                df_resmcc = df_mcc
        return df_resdet, df_resseg, df_resmt, df_resmcc



def main():



    seg_box = [[1,1,4,4],[5,6,7,6],[3,5,4,6]]
    ref_box = [[1,1,3,3],[6,6,6,6],[3,5,4,5],[4,5,6,6]]
    seg_prob = [0.4, 0.8 , 0.9]
    seg_class = [1, 1, 1]
    ref_class = [1, 1, 2, 1]
    dict_file = {}
    dict_file['pred_loc'] = [seg_box]
    dict_file['ref_loc'] = [ref_box]
    dict_file['pred_prob'] = [seg_prob]
    dict_file['pred_class'] = [seg_class]
    dict_file['ref_class'] = [ref_class]
    dict_file['list_values'] = [1,2]
    f = open('TestDataMR.pkl', 'wb')  # Pickle file is newly created where foo1.py is
    pkl.dump(dict_file, f)  # dump data to f
    f.close()
    PE = ProcessEvaluation('TestDataMR.pkl','Object Detection',localization='iou', assignment='Greedy IoU',measures_overlap=[],measures_mcc=[],measures_pcc=['fbeta'],measures_mt=['ap'],measures_boundary=[])
    df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()
    print(df_resdet, df_resseg, df_resmt)

    pred_ilc = [0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2]
    ref_ilc =  [0,1,0,0,0,0,1,1,1,2,2,1,1,1,0,0,2,2]
    pred_prob = [0.4,0.5,0.5,0.75,0.9,0.8,0.5,0.4,0.9,0.8,0.8,0.7,0.6,0.5,0.6,0.7,0.9,1]
    dict_file = {}
    dict_file['pred_class'] = [pred_ilc]
    dict_file['ref_class'] = [ref_ilc]
    dict_file['pred_prob'] = [pred_prob]
    dict_file['list_values'] = [0,1,2]
    f = open('TestILC.pkl', 'wb')  # Pickle file is newly created where foo1.py is
    pkl.dump(dict_file, f)  # dump data to f
    f.close()

    PE = ProcessEvaluation('TestILC.pkl','Image Classification',localization=None,assignment=None,measures_pcc=['lr+'],measures_mt=['auroc','sens@spec'], measures_mcc=['mcc'])
    df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()
    AS = AssociationMapping(seg_box, ref_box, seg_prob, distance='iou',thresh=0)
    matching, list = AS.resolve_ambiguities_matching()

    rng = default_rng(12345)
    ref = rng.random([240,240,240])
    ref = np.where(ref>=0.5, np.ones_like(ref),np.zeros_like(ref))
    seg = rng.random([240,240,240])
    pe = BinaryPairwiseMeasures(seg, ref, measures_pcc =['accuracy','cohens_kappa','mcc','fbeta','sens@spec'],
                          measures_mt=['ap','auroc'],measures_boundary=['masd','nsd'])
    str_res = pe.to_string_count()
    str_dist = pe.to_string_dist()
    str_mt = pe.to_string_mt()
    test = roc_auc_score(np.reshape(ref,-1),np.reshape(seg,-1))
    print(pe)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
