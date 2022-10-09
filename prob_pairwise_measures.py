import numpy as np

from utils import (
    CacheFunctionOutput, 
    trapezoidal_integration,
    x_at_y,
)


class ProbabilityPairwiseMeasures(object):
    def __init__(
        self,
        pred_proba,
        ref_proba,
        case=None,
        measures=[],
        num_neighbors=8,
        pixdim=[1, 1, 1],
        empty=False,
        dict_args={},
    ):
        self.measures_dict = {
            "sens@ppv": (self.sensitivity_at_ppv, "Sens@PPV"),
            "ppv@sens": (self.ppv_at_sensitivity, "PPV@Sens"),
            "sens@spec": (self.sensitivity_at_specificity, "Sens@Spec"),
            "spec@sens": (self.specificity_at_sensitivity, "Spec@Sens"),
            "fppi@sens": (
                self.fppi_at_sensitivity,
                "FPPI@Sens",
            ),
            "sens@fppi": (self.sensitivity_at_fppi, "Sens@FPPI"),
            "auroc": (self.auroc, "AUROC"),
            "ap": (self.average_precision, "AP"),
            "froc": (self.froc, "FROC"),
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
    def all_multi_threshold_values(
        self, max_number_samples=150, max_number_thresh=1500
    ):
        """
        Function defining the list of values for ppv, sensitivity, specificity 
        and FPPI according to a list of probabilistic thresholds. The thresholds 
        are defined to obtain equal bin sizes
        The default maximum number of thresholds is 1500
        Returns:
            unique_new_thresh, list_sens, list_spec, list_ppv, list_fppi
        """
        hist_counts, hist_edges = np.histogram(self.pred, bins=max_number_thresh)
        ix_nonzero = np.flatnonzero(hist_counts)
        unique_new_thresh = hist_edges[np.flatnonzero(ix_nonzero)]

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
        """
        Map of FP given a specific threshold value
        """
        pred_bin = self.pred >= thresh
        return np.asarray((pred_bin - self.ref) > 0.0, dtype=np.float32)

    def __fn_map_thr(self, thresh):
        """
        This function calculates the false negative map based on a threshold
        :return: FN map
        """
        pred_bin = self.pred >= thresh
        return np.asarray((self.ref - pred_bin) > 0.0, dtype=np.float32)

    def __tp_map_thr(self, thresh):
        """
        TP map given a specified threshold
        """
        pred_bin = self.pred >= thresh
        return np.asarray((self.ref + pred_bin) > 1.0, dtype=np.float32)

    def __tn_map_thr(self, thresh):
        """
        TN map given a specified threshold
        """
        pred_bin = self.pred >= thresh
        return np.asarray((self.ref + pred_bin) < 0.5, dtype=np.float32)

    def positive_predictive_values_thr(self, thresh):
        """
        PPV given a specified threshold
        """
        if self.flag_empty:
            return -1
        return self.tp_thr(thresh) / (self.tp_thr(thresh) + self.fp_thr(thresh))

    def specificity_thr(self, thresh):
        """
        Specificity given a specified threshold
        """
        return self.tn_thr(thresh) / self.n_neg_ref()

    def sensitivity_thr(self, thresh):
        """
        Sensitivity given a specified threshold
        """
        return self.tp_thr(thresh) / self.n_pos_ref()

    def fppi_thr(self, thresh):
        """
        Computes average false positives per image. Assumes images are
        stacked on the last dimension.
        """
        if self.case is not None:
            list_sum = []
            for f in range(np.max(self.case)):
                ind_case = np.where(self.case == f)[0]
                case_tmp = ProbabilityPairwiseMeasures(
                    self.pred[ind_case], self.ref[ind_case]
                )
                list_sum.append(case_tmp.fp_thr(thresh))
            fppi = np.mean(np.asarray(list_sum))
        else:
            sum_per_image = np.sum(
                np.reshape(self.__fp_map_thr(thresh), [-1, self.ref.shape[-1]]), axis=0
            )
            fppi = np.mean(sum_per_image)
        return fppi

    def net_benefit_treated(self):
        """
        Calculation of net benefit given a specified threshold
        """
        if "benefit_proba" in self.dict_args.keys():
            thresh = self.dict_args["benefit_proba"]
        else:
            thresh = 0.5
        tp_thresh = self.tp_thr(thresh)
        fp_thresh = self.fp_thr(thresh)
        n = np.size(np.asarray(self.pred))
        return tp_thresh / n * (fp_thresh / n) * (thresh / (1 - thresh))

    def auroc(self):
        """
        Calculation of AUROC using trapezoidal integration based
        on the threshold and values list obtained from the 
        all_multi_threshold_values method
        """
        _, list_sens, list_spec, _, _ \
            = self.all_multi_threshold_values()
        # False positive rate (FPR)
        x = 1 - np.asarray(list_spec)  # specificity to FPR
        # Sensitivity
        y = np.asarray(list_sens)
        # Compute AUROC
        auroc = trapezoidal_integration(x, y)
        return auroc

    def froc(self):
        """
        Calculation of FROC score
        """
        
        _, list_sens, _, _, list_fppi \
            = self.all_multi_threshold_values()
        # Average false positives per image (FPPI)        
        x = np.asarray(list_fppi)
        # Sensitivity
        y = np.asarray(list_sens)                
        # Compute FROC
        froc = trapezoidal_integration(x, y)
        return froc

    def average_precision(self):
        """
        Average precision calculation using trapezoidal integration
        """
        _, list_sens, _, list_ppv, _ \
            = self.all_multi_threshold_values()
        # Sensitivity  
        x = np.asarray(list_sens)
        # Precision
        y = np.asarray(list_ppv)                
        # Average precision (AP)
        ap = trapezoidal_integration(x, y)
        return ap

    def sensitivity_at_specificity(self, value_spec=0.8):
        """
        From specificity cut-off values in the value_specificity field 
        of the dictionary of arguments dict_args, 
        reading of the maximum sensitivity value for all specificities
        larger than the specified value. If value not specified, 
        calculated at specificity of 0.8
        """
        if "value_specificity" in self.dict_args.keys():
            value_spec = self.dict_args["value_specificity"]
        _, list_sens, list_spec, _, _ \
            = self.all_multi_threshold_values()
        return x_at_y(list_sens, list_spec, value_spec)

    def specificity_at_sensitivity(self, value_sens=0.8):
        """
        Specificity given specified sensitivity (Field value_sensitivity)
        in the arguments dictionary. If not specified, calculated at sensitivity=0.8
        """
        if "value_sensitivity" in self.dict_args.keys():
            value_sens = self.dict_args["value_sensitivity"]
        _, list_sens, list_spec, _, _ \
            = self.all_multi_threshold_values()
        return x_at_y(list_spec, list_sens, value_sens)

    def fppi_at_sensitivity(self, value_sens=0.8):
        """
        FPPI value at specified sensitivity value (Field value_sensitivity)
        in the arguments' dictionary. If not specified, calculated at sensitivity 0.8
        """
        if "value_sensitivity" in self.dict_args.keys():
            value_sens = self.dict_args["value_sensitivity"]
        _, list_sens, _, _, list_fppi \
            = self.all_multi_threshold_values()
        return x_at_y(list_fppi, list_sens, value_sens)

    def sensitivity_at_fppi(self, value_fppi=0.8):
        """
        Sensitivity at specified value of FPPI (Field value_fppi)
        in the argument's dictionary. If not specified calculated at FPPI=0.8
        """
        if "value_fppi" in self.dict_args.keys():
            value_fppi = self.dict_args["value_fppi"]            
        _, list_sens, _, _, list_fppi \
            = self.all_multi_threshold_values()
        return x_at_y(list_sens, list_fppi, value_fppi)

    def sensitivity_at_ppv(self, value_ppv=0.8):
        """
        Sensitivity at specified PPV (field value_ppv) in the
        arguments' dictionary. If not specified, calculated at value 0.8
        """
        if "value_ppv" in self.dict_args.keys():
            value_ppv = self.dict_args["value_ppv"]
        _, list_sens, _, list_ppv, _ \
            = self.all_multi_threshold_values()
        return x_at_y(list_sens, list_ppv, value_ppv)

    def ppv_at_sensitivity(self, value_sens=0.8):
        """
        PPV at specified sensitivity value (Field value_sensitivity)
        in the argument's dictionary. If not specified, calculated at value 0.8
        """
        if "value_sensitivity" in self.dict_args.keys():
            value_sens = self.dict_args["value_sensitivity"]
        _, list_sens, _, list_ppv, _ \
            = self.all_multi_threshold_values()
        return x_at_y(list_ppv, list_sens, value_sens)
