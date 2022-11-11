import numpy as np
#from metrics.pairwise_measures import CacheFunctionOutput
from MetricsReloaded.utility.utils import CacheFunctionOutput,max_x_at_y_more, max_x_at_y_less, min_x_at_y_more, min_x_at_y_less, trapezoidal_integration


__all__ = [
    'CalibrationMeasures',
]

class CalibrationMeasures(object):
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
            "ece": (self.expectation_calibration_error, "ECE"),
            "bs": (self.brier_score, "BS"),
            "ls": (self.logarithmic_score, "LS")
        }

        self.pred = pred_proba
        self.ref = ref_proba
        self.case = case
        self.flag_empty = empty
        self.dict_args = dict_args
        self.measures = measures if measures is not None else self.measures_dict

    def expectation_calibration_error(self):
        if 'bins_ece' in self.dict_args:
            nbins = self.dict_args['bins_ece']
        else:
            nbins = 10
        step = 1.0 / nbins
        range_values = np.arange(0,1.00001,step)
        print(range_values)
        list_values = []
        numb_samples = 0
        for (l,u) in zip(range_values[:-1],range_values[1:]):
            ref_tmp = np.where(np.logical_and(self.pred>l, self.pred<=u),self.ref,np.ones_like(self.ref)*-1)
            ref_sel = ref_tmp[ref_tmp>-1]
            nsamples = np.size(ref_sel)
            prop = np.sum(ref_sel)/nsamples
            pred_tmp = np.where(np.logical_and(self.pred>l, self.pred<=u),self.pred,np.ones_like(self.pred)*-1)
            pred_sel = pred_tmp[pred_tmp>-1]
            if nsamples == 0 :
                list_values.append(0)
            else:
                list_values.append(nsamples * np.abs(prop-np.mean(pred_sel)))
            numb_samples += nsamples
        print(list_values,numb_samples)
        return np.sum(np.asarray(list_values))/numb_samples

    def brier_score(self):
        """ 
        Calculation of the Brier score https://en.wikipedia.org/wiki/Brier_score
        """
        bs = np.mean(np.square(self.ref - self.pred))
        return bs

    def logarithmic_score(self):
        """
        Calculation of the logarithmic score https://en.wikipedia.org/wiki/Scoring_rule
        """
        eps = 1e-10
        log_pred = np.log(self.pred + eps)
        log_1pred = np.log(1-self.pred + eps)
        print(log_pred, log_1pred, self.ref, 1-self.ref)
        overall = self.ref * log_pred + (1-self.ref) * log_1pred
        print(overall)
        ls = np.mean(overall)
        print(ls)
        return ls

    def top_label_classification_error(self):
        """
        Calculation of the top-label classification error. Assumes pred_proba a matrix K x Numb observations 
        with probability to be in class k for observation i in position (k,i)
        """
        class_max = np.argmax(self.pred, 0)
        prob_pred_max = np.max(self.pred, 0)
        prob = np.zeros([self.pred.shape[0]])
        prob_ref_values, prob_ref_counts = np.unique(self.ref, return_counts=True)
        for k in range(self.pred.shape[0]):
            idx = np.where(prob_ref_values==k)
            if len(idx) == 0:
                prob[k] = 0
            else:
                prob[k] = prob_ref_counts[idx[0]]/self.pred.shape[1]
        
        prob_expected_max = prob[class_max]
        print(prob, prob_ref_counts, prob_expected_max, prob_pred_max)
        print(np.square(prob_expected_max-prob_pred_max))
        tce = np.sqrt(np.mean(np.square(prob_expected_max-prob_pred_max)))
        return tce
            

    def class_wise_brier_score(self):
        cwbs = 0
        return cwbs

    def class_wise_calibration_error(self):
        cwce = 0
        return cwce

    def kernel_calibration_error(self):
        """
        Based on the paper Widmann, D., Lindsten, F., and Zachariah, D. 
        Calibration tests in multi-class classification: A unifying framework.
         Advances in Neural Information Processing Systems, 32:12257–12267, 2019.
        """
        kce = 0
        return kce

    def negative_log_likelihood(self):
        nll = 0
        return nll

    def root_brier_score(self):
        """
        Gruber S. and Buettner F., Better Uncertainty Calibration via Proper Scores 
        for Classification and Beyond, In Proceedings of the 36th International 
        Conference on  Neural Information Processing Systems, 2022
        """
        rbs = 0
        return rbs

    