import numpy as np
from pairwise_measures import CacheFunctionOutput


class ProbabilityPairwiseMeasures(object):
    def __init__(self, pred_proba, ref_proba, case=None,
                 measures=[], num_neighbors=8, pixdim=[1, 1, 1],
                 empty=False, dict_args={}):
        self.measures_dict = {
            'sens@ppv': (self.sensitivity_at_ppv, 'Sens@PPV'),
            'ppv@sens': (self.ppv_at_sensitivity, 'PPV@Sens'),
            'sens@spec': (self.sensitivity_at_specificity, 'Sens@Spec'),
            'spec@sens': (self.specificity_at_sensitivity, 'Spec@Sens'),
            'fppi@sens': (self.fppi_at_sensitivity, 'FPPI@Sens',),
            'sens@fppi': (self.sensitivity_at_fppi, 'Sens@FPPI'),

            'auroc': (self.auroc, 'AUROC'),
            'ap': (self.average_precision, 'AP'),
            'froc': (self.froc, 'FROC')
        }

        self.pred = pred_proba
        self.ref = ref_proba
        self.case = case
        self.flag_empty = empty
        self.dict_args = dict_args
        self.measures = measures if measures is not None else self.measures_dict

    @CacheFunctionOutput
    def fp_thr(self, thresh):
        return np.sum(self.__fp_map_thr(thresh))

    @CacheFunctionOutput
    def fn_thr(self, thresh):
        return np.sum(self.__fn_map_thr(thresh))
    
    @CacheFunctionOutput
    def tp_thr(self, thresh):
        return np.sum(self.__tp_map_thr(thresh))

    @CacheFunctionOutput
    def tn_thr(self, thresh):
        return np.sum(self.__tn_map_thr(thresh))

    @CacheFunctionOutput
    def n_pos_ref(self):
        return np.sum(self.ref)

    @CacheFunctionOutput
    def n_neg_ref(self):
        return np.sum(1 - self.ref)
    
    
    @CacheFunctionOutput
    def all_multi_threshold_values(self, max_number_samples=150, max_number_thresh=1500):
        unique_thresh, unique_counts = np.unique(self.pred, return_counts=True)
        if len(unique_thresh) < max_number_thresh:
            unique_new_thresh = unique_thresh
        elif np.size(self.ref) < max_number_samples:
            unique_new_thresh = unique_thresh
        else:
            numb_thresh_temp = np.size(self.ref) / max_number_samples
            numb_samples_temp = np.size(self.pred) / max_number_thresh

            unique_new_thresh = [0]
            current_count = 0
            for (f, c) in zip(unique_thresh, unique_counts):
                if current_count < numb_samples_temp:
                    current_count += c
                    new_thresh = f
                else:
                    unique_new_thresh.append(new_thresh)
                    current_count = 0
        list_sens = []
        list_spec = []
        list_ppv = []
        list_fppi = []
        unique_new_thresh = np.sort(unique_new_thresh)[::-1]
        for val in unique_new_thresh:
            list_sens.append(self.sensitivity_thr(val))
            list_spec.append(self.specificity_thr(val))
            list_ppv.append(self.positive_predictive_values_thr(val))
            list_fppi.append(self.fppi_thr(val))
        return unique_new_thresh, list_sens, list_spec, list_ppv, list_fppi

    def __fp_map_thr(self, thresh):
        pred_bin = self.pred >= thresh
        return np.asarray((pred_bin-self.ref)>0.0, dtype=np.float32)

    def __fn_map_thr(self, thresh):
        """
        This function calculates the false negative map based on a threshold
        :return: FN map
        """
        pred_bin = self.pred >= thresh
        return np.asarray((self.ref - pred_bin) > 0.0, dtype=np.float32)

    def __tp_map_thr(self, thresh):
        pred_bin = self.pred >= thresh
        return np.asarray((self.ref+pred_bin)>1.0, dtype=np.float32)

    def __tn_map_thr(self, thresh):
        pred_bin = self.pred >= thresh
        return np.asarray((self.ref+pred_bin)<0.5, dtype=np.float32)

    def positive_predictive_values_thr(self, thresh):
        if self.flag_empty:
            return -1
        return self.tp_thr(thresh) / (self.tp_thr(thresh) + self.fp_thr(thresh))

    def specificity_thr(self, thresh):
        return self.tn_thr(thresh) / self.n_neg_ref()

    def sensitivity_thr(self, thresh):
        return self.tp_thr(thresh) / self.n_pos_ref()

    def recall_thr(self, thresh):
        return self.tp_thr(thresh) / (self.tp_thr(thresh) + self.fn_thr(thresh))

    def fppi_thr(self, thresh):
        if self.case is not None:
            list_sum = []
            for f in range(np.max(self.case)):
                ind_case = np.where(self.case==f)[0]
                case_tmp = ProbabilityPairwiseMeasures(self.pred[ind_case], self.ref[ind_case])
                list_sum.append(case_tmp.fp_thr(thresh))
            fppi = np.mean(np.asarray(list_sum))
        else:
            sum_per_image = np.sum(np.reshape(self.__fp_map_thr(thresh), [-1, self.ref.shape[-1]]), axis=0)
            fppi = np.mean(sum_per_image)
        return fppi

    def net_benefit_treated(self):
        if 'benefit_proba' in self.dict_args.keys():
            thresh = self.dict_args['benefit_proba']
        else:
            thresh = 0.5
        tp_thresh = self.tp_thr(thresh)
        fp_thresh = self.fp_thr(thresh)
        n = np.size(np.asarray(self.pred))
        return tp_thresh/n * (fp_thresh/n) * (thresh/(1-thresh))

    def auroc(self):
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_spec = np.asarray(list_spec)
        array_sens = np.asarray(list_sens)
        diff_spec = (1 - array_spec[1:]) - (1 - array_spec[:-1])
        diff_sens = array_sens[1:] - array_sens[:-1]
        bottom_rect = np.sum(array_sens[:-1] * diff_spec)
        top_rect = np.sum(array_sens[1:] * diff_spec)
        diff_rect = np.sum(diff_sens * diff_spec)
        auroc = bottom_rect + diff_rect * 0.5
        return auroc

    def froc(self):
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_fppi = np.asarray(list_fppi)
        array_sens = np.asarray(list_sens)
        diff_fppi = array_fppi[1:] - array_fppi[:-1]
        diff_sens = array_sens[1:] - array_sens[:-1]
        bottom_rect = np.sum(array_sens[:-1] * diff_fppi)
        top_rect = np.sum(array_sens[1:] * diff_fppi)
        diff_rect = np.sum(diff_sens * diff_fppi)
        froc = bottom_rect + diff_rect * 0.5
        return froc

    def average_precision(self):
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        diff_ppv = np.asarray(list_ppv[1:]) - np.asarray(list_ppv[:-1])
        diff_sens = np.asarray(list_sens[1:]) - np.asarray(list_sens[:-1])
        bottom_rect = np.sum(np.asarray(list_ppv[:-1]) * diff_sens)
        top_rect = np.sum(np.asarray(list_ppv[1:]) * diff_sens)
        diff_rect = np.sum(diff_sens * diff_ppv)
        ap = bottom_rect + diff_rect * 0.5
        return ap

    def sensitivity_at_specificity(self):
        if 'value_specificity' in self.dict_args.keys():
            value_spec = self.dict_args['value_specificity']
        else:
            value_spec = 0.8
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_spec = np.asarray(list_spec)
        ind_values = np.where(array_spec >= value_spec)
        array_sens = np.asarray(list_sens)
        sens_valid = array_sens[ind_values]
        return np.max(sens_valid)

    def specificity_at_sensitivity(self):
        if 'value_sensitivity' in self.dict_args.keys():
            value_sens = self.dict_args['value_sensitivity']
        else:
            value_sens = 0.8
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_spec = np.asarray(list_spec)
        array_sens = np.asarray(list_sens)
        ind_values = np.where(array_sens >= value_sens)
        spec_valid = array_spec[ind_values]
        return np.max(spec_valid)

    def fppi_at_sensitivity(self):
        if 'value_sensitivity' in self.dict_args.keys():
            value_sens = self.dict_args['value_sensitivity']
        else:
            value_sens = 0.8
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_fppi = np.asarray(list_fppi)
        array_sens = np.asarray(list_sens)
        ind_values = np.where(array_sens >= value_sens)
        fppi_valid = array_fppi[ind_values]
        return np.max(fppi_valid)

    def sensitivity_at_fppi(self):
        if 'value_fppi' in self.dict_args.keys():
            value_fppi = self.dict_args['value_fppi']
        else:
            value_fppi = 0.8
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_fppi = np.asarray(list_fppi)
        array_sens = np.asarray(list_sens)
        ind_values = np.where(array_fppi <= value_fppi)
        sens_valid = array_sens[ind_values]
        return np.max(sens_valid)

    def sensitivity_at_ppv(self):
        if 'value_ppv' in self.dict_args.keys():
            value_ppv = self.dict_args['value_ppv']
        else:
            value_ppv = 0.8
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_ppv = np.asarray(list_ppv)
        array_sens = np.asarray(list_sens)
        ind_values = np.where(array_ppv >= value_ppv)
        sens_valid = array_sens[ind_values]
        return np.max(sens_valid)

    def ppv_at_sensitivity(self):
        if 'value_sensitivity' in self.dict_args.keys():
            value_sens = self.dict_args['value_sensitivity']
        else:
            value_sens = 0.8
        unique_thresh, list_sens, list_spec, list_ppv, list_fppi = self.all_multi_threshold_values()
        array_ppv = np.asarray(list_ppv)
        array_sens = np.asarray(list_sens)
        ind_values = np.where(array_sens >= value_sens)
        ppv_valid = array_ppv[ind_values]
        return np.max(ppv_valid)

    def to_dict_meas(self, fmt='{:.4f}'):
        result_dict = {}
        # list_space = ['com_ref', 'com_pred', 'list_labels']
        for key in self.measures:
            result = self.measures_dict[key][0]()
            result_dict[key] = fmt.format(result)
        return result_dict  # trim the last comma

