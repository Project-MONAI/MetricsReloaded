import numpy as np
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM
from MetricsReloaded.processes.mixed_measures_processes import MultiLabelPairwiseMeasures as MLPM
from MetricsReloaded.processes.overall_process import ProcessEvaluation as PE
from matplotlib import pyplot as plt

# Code for setting of a set of predictions
pq_pred1 = np.zeros([21, 21])
pq_pred1[5:7, 2:5] = 1
pq_pred2 = np.zeros([21, 21])
pq_pred2[14:18, 4:6] = 1
pq_pred2[16, 3] = 1
pq_pred3 = np.zeros([21, 21])
pq_pred3[14:18, 7:12] = 1
pq_pred4 = np.zeros([21, 21])
pq_pred4[2:8, 13:16] = 1
pq_pred4[2:4, 12] = 1


# Code for setting of a set of references
pq_ref1 = np.zeros([21, 21])
pq_ref1[8:11, 3] = 1
pq_ref1[9, 2:5] = 1
pq_ref2 = np.zeros([21, 21])
pq_ref2[14:19, 7:13] = 1
pq_ref3 = np.zeros([21, 21])
pq_ref3[2:7, 14:17] = 1
pq_ref3[2:4, 12:14] = 1




prediction = pq_pred1 + pq_pred2 + pq_pred3 + pq_pred4
reference = pq_ref1 + pq_ref2 + pq_ref3

print('Creation of dictionary for one single comparison case with two metrics')
bpm = BPM(prediction, reference, measures=['fbeta','nsd'])
dict_seg = bpm.to_dict_meas()
print(dict_seg)

print("Direct call to the metric to calculate")
bpm.fbeta()

# Combining multiple np arrays so as to compare item i of predictions with item i of references

# list of reference np arrays
list_ref = [pq_ref1, pq_ref2, pq_ref3]
# list of prediction np arrays
list_pred = [pq_pred1, pq_pred3, pq_pred4]
# No use of probability maps in this setting so 
list_prob = [None, None, None]

print('Creation of process for multiple cases')
mlpm = MLPM(list_pred, list_ref,list_prob,list_values=[1],measures_pcc=['fbeta','nsd'],per_case=True)
df_seg, df_mt = mlpm.per_label_dict()
print(df_seg)

# Showing what happens when there is an error in input: here asking to calculate auroc without a probabilistic map
print("Creation of multi process with error in choice of measure due to absence of probabilistic input")
mlpm = MLPM(list_pred, list_ref,list_prob,list_values=[1],measures_pcc=['fbeta'],measures_boundary=['nsd'],measures_mt=['auroc'],per_case=True, pixdim=[1,1])
df_seg, df_mt = mlpm.per_label_dict()
print(df_seg)

# How to implement the full process when considering the three lists and resulting in a PE object 
print("Creation of the full process")
data = {}
data['pred_class'] = list_pred
data['ref_class'] = list_ref
data['pred_prob'] = list_prob
data['list_values'] = [1] # Only one labelling - binary choice here
pe = PE(data, 'Semantic Segmentation',measures_overlap=['fbeta'],measures_boundary=['nsd'],case=True)

print(pe.resseg)
