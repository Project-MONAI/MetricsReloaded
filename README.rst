================
Metrics Reloaded
================

.. start-description

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


A Python implementaiton of `Metrics Reloaded <https://openreview.net/forum?id=24kBqy8rcB_>`__ - A new recommendation framework for biomedical image analysis validation.

.. figure:: /images/classification_scales_and_domains.png
    :scale: 10%
    :align: center

    Metrics Reloaded fosters the convergence of validation methodology across modalities, application domains and classification scales

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

.. end-description
