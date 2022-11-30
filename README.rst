================
Metrics Reloaded
================

.. start-description

A Python implementaiton of `Metrics Reloaded <https://openreview.net/forum?id=24kBqy8rcB_>`__ - A new recommendation framework for biomedical image analysis validation.

.. start-badges

|docs|
|testing|
|codecov|

.. |docs| image:: https://readthedocs.org/projects/metricsreloaded/badge/?style=flat
    :target: https://MetricsReloaded.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |testing| image:: https://github.com/csudre/MetricsReloaded/actions/workflows/python-app.yml/badge.svg
    :target: https://github.com/csudre/MetricsReloaded/actions
    :alt: Testing Status

.. |codecov| image:: https://codecov.io/gh/csudre/MetricsReloaded/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/csudre/MetricsReloaded
    :alt: Coverage Status

.. end-badges

Installation
============
Using git
---------

Create and activate a new `Conda <https://docs.conda.io/en/latest/miniconda.html>`__ environment: ::

    conda create -n metrics python=3.10 pip
    conda activate metrics

Clone the repository: ::

    git clone https://github.com/csudre/MetricsReloaded.git
    cd MetricsReloaded

Install the package:

    python -m pip install .

You can alternatively install the package in editable mode:

    python -m pip install -e .

This is useful if you are developing MetricsReloaded and want to see changes in the code automatically applied to the installed library.

With MONAI support
---------

Install the package as:

    python -m pip install .[monai]

to ensure that the MONAI dependency is installed.

The MetricsReloaded metrics can then be used in, e.g., a MONAI training script as::

    from MetricsReloaded.metrics.monai_wrapper import (
        BinaryMetric4Monai,
        CategoricalMetric4Monai,
    )

    # Use binary pair-wise metrics
    metric_name = "Cohens Kappa"
    metric = BinaryMetric4Monai(metric_name=metric_name)
    metric(y_pred=y_pred, y=y)
    value = metric.aggregate().item()

    # Use categorical pair-wise metric
    metric_name = "Matthews Correlation Coefficient"
    metric = CategoricalMetric4Monai(metric_name=metric_name)
    metric(y_pred=y_pred, y=y)
    value = metric.aggregate().item()

Overview
========

.. end-description

.. figure:: docs/source/images/classification_scales_and_domains.png
    :scale: 10%
    :align: center

    Metrics Reloaded fosters the convergence of validation methodology across modalities, application domains and classification scales
