import pytest
from metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
from processes.mixed_measures_processes import MultiLabelLocSegPairwiseMeasure as MLIS
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import matthews_corrcoef as mcc

from metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures
