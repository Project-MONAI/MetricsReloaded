================
Metrics Reloaded
================

.. start-description

A Python implementation of `Metrics Reloaded <https://openreview.net/forum?id=24kBqy8rcB_>`__ - A new recommendation framework for biomedical image analysis validation.

ℹ️ This is a fork of the `Project-MONAI/MetricsReloaded <https://github.com/Project-MONAI/MetricsReloaded>`__ repo. In this fork, we have added some additional metrics, such as `relative volume error (RVE) <https://github.com/ivadomed/MetricsReloaded/blob/713892a053e23a40a8bd88aa72a261409d536ba8/MetricsReloaded/metrics/pairwise_measures.py#L897>`__, `lesion-wise F1 Score <https://github.com/ivadomed/MetricsReloaded/blob/713892a053e23a40a8bd88aa72a261409d536ba8/MetricsReloaded/metrics/pairwise_measures.py#L1227>`__, `lesion-wise Sensitivity <https://github.com/ivadomed/MetricsReloaded/blob/713892a053e23a40a8bd88aa72a261409d536ba8/MetricsReloaded/metrics/pairwise_measures.py#L1273>`__, and `lesion-wise positive predictive value (PPV) <https://github.com/ivadomed/MetricsReloaded/blob/713892a053e23a40a8bd88aa72a261409d536ba8/MetricsReloaded/metrics/pairwise_measures.py#L1252>`__.

ℹ️ You can use the `compute_metrics_reloaded.py wrapper script <https://github.com/ivadomed/utilities/blob/main/compute_metrics/compute_metrics_reloaded.py>`__ to compute metrics using the MetricsReloaded package. For the installation and usage of the script, see `our quick start guide <./MetricsReloaded_quick_start_guide.md>`__.

Installation
============
Using git
---------

Create and activate a new `Conda <https://docs.conda.io/en/latest/miniconda.html>`__ environment: ::

    conda create -n metrics python=3.10 pip
    conda activate metrics

Clone the repository: ::

    git clone https://github.com/ivadomed/MetricsReloaded
    cd MetricsReloaded

Install the package:

    python -m pip install .

You can alternatively install the package in editable mode:

    python -m pip install -e .

This is useful if you are developing MetricsReloaded and want to see changes in the code automatically applied to the installed library.


Overview
========

All functions used in this framework are documented `here <https://metricsreloaded.readthedocs.io/en/latest/?badge=latest>`

The repository is organised in three main folders:

- processes: this allows for the combination of multiple metrics in an evaluation setting and reflects the tasks tackled in the MetricsReloaded framework namely:

  #. Image Level Classification (ILC)
  #. Semantic Segmentation (SS)
  #. Object Detection (OD)
  #. Instance Segmentation (SS)

- metrics: this contains all the individual metrics reported and discussed in the MetricsReloaded guidelines. Those are classified as either:

  #. pairwise_measures - all metrics considering a binary or multiclass input
  #. prob_pairwise_measures - all metrics relying on multi threshold and/or probabilistic input
  #. calibration_measures - all metrics related to the evaluation of the calibration of probabilistic outputs

- utility: this contains all ancillary function relevant notably for aggregation of metrics, or preliminary tools required for complext assignments prior to metrics calculation notably in the case of Object Detection and Instance Segmentation. 

Useful links to get started
===========================

To see examples on how to process different cases of tasks please look into the 

:example: `example_ss.py`

Pictorial representation
========================

.. end-description

.. figure:: docs/source/images/classification_scales_and_domains.png
    :scale: 10%
    :align: center

    Metrics Reloaded fosters the convergence of validation methodology across modalities, application domains and classification scales

Support
========================
For any questions or remarks, please contact metrics-reloaded-package(at)dkfz-heidelberg.de.


