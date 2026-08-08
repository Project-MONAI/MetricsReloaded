"""
Uncertainty quantification - :mod:`MetricsReloaded.utility.uncertainty`
=======================================================================

This module provides functions for reporting the uncertainty of aggregated
metric values.

A mean metric over N cases is a point estimate: a mean DSC of 0.85 over 20
cases and over 2,000 cases support very different conclusions, and validation
studies routinely compare methods whose intervals overlap entirely. The
Metrics Reloaded recommendations call for reporting variability alongside
aggregates; this module provides the standard non-parametric tool for it.

.. currentmodule:: MetricsReloaded.utility.uncertainty

.. autosummary::
    :nosignatures:

    percentile_bootstrap_ci
    stats_with_ci

"""

import numpy as np
import pandas as pd

__all__ = [
    "percentile_bootstrap_ci",
    "stats_with_ci",
]


def percentile_bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Percentile-bootstrap confidence interval for the mean of per-case
    metric values.

    NaN values are ignored, consistent with the masked aggregations used
    elsewhere in this package. The resampling is seeded by default so that
    reported intervals are reproducible across runs of the same evaluation.

    :param values: iterable of per-case metric values (may contain NaN)
    :param n_boot: number of bootstrap resamples
    :param alpha: 1 - confidence level (0.05 -> 95% interval)
    :param seed: seed for the resampling generator; pass None for
        non-deterministic resampling
    :return: (lower, upper) bounds of the interval; (nan, nan) when fewer
        than two non-NaN values are available
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[indices].mean(axis=1)
    return (
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def stats_with_ci(df, n_boot=2000, alpha=0.05, seed=42):
    """Summary statistics of a per-case results dataframe with
    confidence-interval rows for each numeric metric column.

    Returns the ``df.describe()`` table augmented with two rows,
    ``ci95_low`` and ``ci95_high`` (names follow the requested ``alpha``),
    holding the percentile-bootstrap interval for each column's mean.

    :param df: dataframe of per-case metric values (columns = metrics)
    :param n_boot: number of bootstrap resamples per column
    :param alpha: 1 - confidence level
    :param seed: seed for reproducible intervals
    :return: describe() dataframe with appended CI rows
    """
    described = df.describe()
    level = int(round((1 - alpha) * 100))
    low_name = "ci%d_low" % level
    high_name = "ci%d_high" % level
    lows = {}
    highs = {}
    for col in described.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        low, high = percentile_bootstrap_ci(
            series.to_numpy(), n_boot=n_boot, alpha=alpha, seed=seed
        )
        lows[col] = low
        highs[col] = high
    described.loc[low_name] = pd.Series(lows)
    described.loc[high_name] = pd.Series(highs)
    return described
