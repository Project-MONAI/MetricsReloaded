from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM
from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures as PPM
from MetricsReloaded.processes.mixed_measures_processes import MultiLabelPairwiseMeasures as MLPM
from MetricsReloaded.processes.overall_process import ProcessEvaluation as PE
import numpy as np
from matplotlib import pyplot as plt



ref = np.asarray([0, 0, 0, 1, 1, 1])
pred = np.asarray([0,0,1,1,0,1])
pred_proba = np.asarray([0.21, 0.35, 0.63, 0.92, 0.32, 0.79])


print('Creation of dictionary for one single comparison case with two metrics')
bpm = BPM(pred, ref, measures=['fbeta','mcc'])
dict_seg = bpm.to_dict_meas()
print(dict_seg)

print("Direct call to the metric to calculate")
bpm.fbeta()


print("Creation of multi process with error in choice of measure due to absence of probabilistic input")
mlpm = MLPM([pred], [ref],[pred_proba],list_values=[1],measures_pcc=['fbeta','mcc'],measures_mt=['auroc'],per_case=True)
df_seg, df_mt = mlpm.per_label_dict()
print(df_seg)


print("Creation of full process")
data = {}
data['pred_class'] = [pred]
data['ref_class'] = [ref]
data['pred_prob'] = [pred_proba]
data['list_values'] = [1]
pe = PE(data, 'ImLC',measures_overlap=['fbeta','mcc'],measures_mt=['auroc'],case=True)

print(pe.resseg)
