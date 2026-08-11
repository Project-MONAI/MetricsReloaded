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
    :param n_boot: number of bootstrap resamples; must be at least 2, and in
        practice should be far larger. Both bounds are quantiles of the
        resample distribution, so while n_boot < 2 / alpha they are decided by
        its most extreme draws alone (fewer than 40 resamples at the default
        alpha=0.05) and the interval is too narrow rather than merely noisy.
    :param alpha: 1 - confidence level (0.05 -> 95% interval); must be
        strictly between 0 and 1
    :param seed: seed for the resampling generator; pass None for
        non-deterministic resampling
    :return: (lower, upper) bounds of the interval; (nan, nan) when fewer
        than two non-NaN values are available
    """
    # n_boot=1 passes any "is it positive?" check and then returns a
    # zero-width interval, because both quantiles of a single resample are
    # that resample: a claim of exact precision produced by the least
    # evidence the function accepts.
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2, got %r" % (n_boot,))
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1, got %r" % (alpha,))
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


def stats_with_ci(df, n_boot=2000, alpha=0.05, seed=42, exclude=()):
    """Summary statistics of a per-case results dataframe with
    confidence-interval rows for each numeric metric column.

    Returns the ``df.describe()`` table augmented with two rows,
    ``ci95_low`` and ``ci95_high`` (names follow the requested ``alpha``;
    fractional levels are preserved, e.g. ``alpha=0.025`` -> ``ci97.5_low``),
    holding the percentile-bootstrap interval for each column's mean.

    :param df: dataframe of per-case metric values (columns = metrics)
    :param n_boot: number of bootstrap resamples per column
    :param alpha: 1 - confidence level
    :param seed: seed for reproducible intervals
    :param exclude: column names to keep in the describe() table but skip
        when bootstrapping (identifier columns such as ``case``); their CI
        cells are left empty
    :return: describe() dataframe with appended CI rows
    """
    described = df.describe()
    level = (1 - alpha) * 100
    low_name = "ci%g_low" % level
    high_name = "ci%g_high" % level
    lows = {}
    highs = {}
    for col in described.columns:
        if col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        low, high = percentile_bootstrap_ci(
            series.to_numpy(), n_boot=n_boot, alpha=alpha, seed=seed
        )
        lows[col] = low
        highs[col] = high
    described.loc[low_name] = pd.Series(lows)
    described.loc[high_name] = pd.Series(highs)
    return described
