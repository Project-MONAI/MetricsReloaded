#######################################################################
#
# Tests for the `compute_metrics_reloaded.py` script
#
# RUN BY:
#   python -m unittest tests/test_pairwise_measures_neuropoly.py
#
# Authors: NeuroPoly team
#
#######################################################################

import unittest
import os
import numpy as np
import nibabel as nib
from compute_metrics_reloaded import compute_metrics_single_subject
import tempfile

METRICS = ['dsc', 'fbeta', 'nsd', 'vol_diff', 'rel_vol_error', 'lesion_ppv', 'lesion_sensitivity', 'lesion_f1_score']


class TestComputeMetricsReloaded(unittest.TestCase):
    def setUp(self):
        # Use tempfile.NamedTemporaryFile to create temporary nifti files
        self.ref_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        self.pred_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        self.metrics = METRICS

    def create_dummy_nii(self, file_obj, data):
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, file_obj.name)
        file_obj.seek(0)  # Move back to the beginning of the file

    def tearDown(self):
        # Close and remove temporary files
        self.ref_file.close()
        os.unlink(self.ref_file.name)
        self.pred_file.close()
        os.unlink(self.pred_file.name)

    def assert_metrics(self, metrics_dict, expected_metrics):
        for metric in self.metrics:
            # Loop over labels/classes (e.g., 1, 2, ...)
            for label in expected_metrics.keys():
                # if value is nan, use np.isnan to check
                if np.isnan(expected_metrics[label][metric]):
                    self.assertTrue(np.isnan(metrics_dict[label][metric]))
                # if value is inf, use np.isinf to check
                elif np.isinf(expected_metrics[label][metric]):
                    self.assertTrue(np.isinf(metrics_dict[label][metric]))
                else:
                    self.assertAlmostEqual(metrics_dict[label][metric], expected_metrics[label][metric])

    def test_empty_ref_and_pred(self):
        """
        Empty reference and empty prediction
        """

        expected_metrics = {1.0: {'EmptyPred': True,
                                  'EmptyRef': True,
                                  'dsc': 1,
                                  'fbeta': 1,
                                  'nsd': np.nan,
                                  'rel_vol_error': 0,
                                  'vol_diff': np.nan,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0}}

        # Create empty reference
        self.create_dummy_nii(self.ref_file, np.zeros((10, 10, 10)))
        # Create empty prediction
        self.create_dummy_nii(self.pred_file, np.zeros((10, 10, 10)))
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_empty_ref(self):
        """
        Empty reference and non-empty prediction
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': True,
                                  'dsc': 0.0,
                                  'fbeta': 0,
                                  'nsd': 0.0,
                                  'rel_vol_error': 100,
                                  'vol_diff': np.inf,
                                  'lesion_ppv': 0.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 0.0}}

        # Create empty reference
        self.create_dummy_nii(self.ref_file, np.zeros((10, 10, 10)))
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[5:7, 2:5] = 1
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_empty_pred(self):
        """
        Non-empty reference and empty prediction
        """

        expected_metrics = {1.0: {'EmptyPred': True,
                                  'EmptyRef': False,
                                  'dsc': 0.0,
                                  'fbeta': 0,
                                  'nsd': 0.0,
                                  'rel_vol_error': -100.0,
                                  'vol_diff': 1.0,
                                  'lesion_ppv': 0.0,
                                  'lesion_sensitivity': 0.0,
                                  'lesion_f1_score': 0.0}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[5:7, 2:5] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create empty prediction
        self.create_dummy_nii(self.pred_file, np.zeros((10, 10, 10)))
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred(self):
        """
        Non-empty reference and non-empty prediction with partial overlap
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': False,
                                  'dsc': 0.26666666666666666,
                                  'fbeta': 0.26666667461395266,
                                  'nsd': 0.5373134328358209,
                                  'rel_vol_error': 300.0,
                                  'vol_diff': 3.0,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[4:5, 3:6] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[4:8, 2:5] = 1
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred_multi_class(self):
        """
        Non-empty reference and non-empty prediction with partial overlap
        Multi-class (i.e., voxels with values 1 and 2, e.g., region-based nnUNet training)
        """

        expected_metrics = {1.0: {'dsc': 0.25,
                                  'fbeta': 0.2500000055879354,
                                  'nsd': 0.5,
                                  'vol_diff': 2.0,
                                  'rel_vol_error': 200.0,
                                  'EmptyRef': False,
                                  'EmptyPred': False,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0},
                            2.0: {'dsc': 0.26666666666666666,
                                  'fbeta': 0.26666667461395266,
                                  'nsd': 0.5373134328358209,
                                  'vol_diff': 3.0,
                                  'rel_vol_error': 300.0,
                                  'EmptyRef': False,
                                  'EmptyPred': False,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[4:5, 3:10] = 1
        ref[4:5, 3:6] = 2  # e.g., lesion within spinal cord
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[4:8, 2:8] = 1
        pred[4:8, 2:5] = 2  # e.g., lesion within spinal cord
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred_with_full_overlap(self):
        """
        Non-empty reference and non-empty prediction with full overlap
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': False,
                                  'dsc': 1.0,
                                  'fbeta': 1.0,
                                  'nsd': 1.0,
                                  'rel_vol_error': 0.0,
                                  'vol_diff': 0.0,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[4:8, 2:5] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[4:8, 2:5] = 1
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)


if __name__ == '__main__':
    unittest.main()
