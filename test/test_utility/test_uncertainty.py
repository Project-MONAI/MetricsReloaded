import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from MetricsReloaded.utility.uncertainty import (
    percentile_bootstrap_ci,
    stats_with_ci,
)


def test_interval_brackets_the_mean_and_is_deterministic():
    values = np.array([0.7, 0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.79, 0.81, 0.86])
    low, high = percentile_bootstrap_ci(values)
    assert low < values.mean() < high
    # seeded resampling: identical call gives identical interval
    assert (low, high) == percentile_bootstrap_ci(values)


def test_interval_narrows_with_more_cases():
    rng = np.random.default_rng(0)
    small = rng.normal(0.8, 0.1, size=20)
    large = np.concatenate([small] * 50)
    low_s, high_s = percentile_bootstrap_ci(small)
    low_l, high_l = percentile_bootstrap_ci(large)
    assert (high_l - low_l) < (high_s - low_s)


def test_approximates_analytic_interval_for_gaussian_data():
    rng = np.random.default_rng(1)
    values = rng.normal(0.5, 0.2, size=400)
    low, high = percentile_bootstrap_ci(values, n_boot=5000)
    sem = values.std(ddof=1) / np.sqrt(values.size)
    assert_allclose(low, values.mean() - 1.96 * sem, atol=3 * sem / 10)
    assert_allclose(high, values.mean() + 1.96 * sem, atol=3 * sem / 10)


def test_nan_values_are_ignored():
    values = [0.8, np.nan, 0.9, 0.85, np.nan, 0.82]
    low, high = percentile_bootstrap_ci(values)
    clean_low, clean_high = percentile_bootstrap_ci([0.8, 0.9, 0.85, 0.82])
    assert (low, high) == (clean_low, clean_high)


def test_degenerate_inputs_return_nan_interval():
    assert np.isnan(percentile_bootstrap_ci([])[0])
    assert np.isnan(percentile_bootstrap_ci([0.5])[1])
    assert np.isnan(percentile_bootstrap_ci([np.nan, np.nan])[0])


def test_stats_with_ci_appends_interval_rows():
    df = pd.DataFrame(
        {
            "dsc": [0.7, 0.8, 0.85, 0.9, 0.75],
            "nsd": [0.6, 0.65, 0.7, 0.72, 0.68],
        }
    )
    stats = stats_with_ci(df)
    assert "ci95_low" in stats.index
    assert "ci95_high" in stats.index
    for col in ("dsc", "nsd"):
        assert stats.loc["ci95_low", col] < df[col].mean()
        assert stats.loc["ci95_high", col] > df[col].mean()
    # describe() content preserved
    assert_allclose(stats.loc["mean", "dsc"], df["dsc"].mean())


def test_invalid_options_are_rejected():
    values = [0.7, 0.8, 0.9]
    with pytest.raises(ValueError):
        percentile_bootstrap_ci(values, n_boot=0)
    with pytest.raises(ValueError):
        percentile_bootstrap_ci(values, alpha=1.1)
    with pytest.raises(ValueError):
        percentile_bootstrap_ci(values, alpha=0.0)


def test_single_resample_is_rejected_rather_than_reported_as_exact():
    # Both bounds are quantiles of the resample distribution, so with one
    # resample they are the same number and the interval has zero width --
    # the strongest possible claim from the weakest possible evidence.
    with pytest.raises(ValueError):
        percentile_bootstrap_ci([0.1, 0.5, 0.9, 0.3, 0.7], n_boot=1)


def test_fractional_confidence_level_is_labelled_exactly():
    df = pd.DataFrame({"dsc": [0.7, 0.8, 0.85, 0.9, 0.75]})
    stats = stats_with_ci(df, alpha=0.025)
    # a 97.5% interval must not be rounded to "ci98"
    assert "ci97.5_low" in stats.index
    assert "ci97.5_high" in stats.index


def test_identifier_columns_are_excluded_from_bootstrapping():
    df = pd.DataFrame(
        {
            "case": [1, 2, 3, 4, 5],
            "dsc": [0.7, 0.8, 0.85, 0.9, 0.75],
        }
    )
    stats = stats_with_ci(df, exclude=("case",))
    # the identifier keeps its describe() rows but gets no interval
    assert_allclose(stats.loc["mean", "case"], 3.0)
    assert np.isnan(stats.loc["ci95_low", "case"])
    assert np.isnan(stats.loc["ci95_high", "case"])
    # metric columns still get one
    assert stats.loc["ci95_low", "dsc"] < df["dsc"].mean()
