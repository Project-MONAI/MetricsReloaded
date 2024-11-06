from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM
from MetricsReloaded.processes.mixed_measures_processes import MultiLabelPairwiseMeasures as MLPM
from MetricsReloaded.processes.overall_process import ProcessEvaluation as PE
import numpy as np
from matplotlib import pyplot as plt

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
prediction = pq_pred1 + pq_pred2 + pq_pred3 + pq_pred4

pq_ref1 = np.zeros([21, 21])
pq_ref1[8:11, 3] = 1
pq_ref1[9, 2:5] = 1
pq_ref2 = np.zeros([21, 21])
pq_ref2[14:19, 7:13] = 1
pq_ref3 = np.zeros([21, 21])
pq_ref3[2:7, 14:17] = 1
pq_ref3[2:4, 12:14] = 1
reference = pq_ref1 + pq_ref2 + pq_ref3




list_ref = [pq_ref1, pq_ref2, pq_ref3]
list_pred = [pq_pred1, pq_pred3, pq_pred4]
list_prob = [None, None, None]

print('Creation of dictionary for one single comparison case with two metrics')
bpm = BPM(prediction, reference, measures=['fbeta','nsd'])
dict_seg = bpm.to_dict_meas()
print(dict_seg)

print("Direct call to the metric to calculate")
bpm.fbeta()

print('Creation of process for multiple cases')
mlpm = MLPM(list_pred, list_ref,list_prob,list_values=[1],measures_pcc=['fbeta','nsd'],per_case=True)
df_seg, df_mt = mlpm.per_label_dict()
print(df_seg)

print("Creation of multi process with error in choice of measure due to absence of probabilistic input")
mlpm = MLPM(list_pred, list_ref,list_prob,list_values=[1],measures_pcc=['fbeta'],measures_boundary=['nsd'],measures_mt=['auroc'],per_case=True, pixdim=[1,1])
df_seg, df_mt = mlpm.per_label_dict()
print(df_seg)


print("Creation of full process")
data = {}
data['pred_class'] = list_pred
data['ref_class'] = list_ref
data['pred_prob'] = list_prob
data['list_values'] = [1]
pe = PE(data, 'SemS',measures_overlap=['fbeta'],measures_boundary=['nsd'],case=True)

print(pe.resseg)
