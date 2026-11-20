"""
Tests for the density estimators and bandwidth selectors (no plotting)
"""

import warnings

import numpy as np
import pytest
from scipy import stats

from plotnine_ggdist import (
    bandwidth_dpi,
    bandwidth_nrd,
    bandwidth_nrd0,
    bounder_cdf,
    bounder_cooke,
    bounder_range,
    density_bounded,
    density_histogram,
    density_unbounded,
    weighted_ecdf,
    weighted_quantile,
)

rng = np.random.default_rng(7)
X = rng.normal(size=2000)
BETA = rng.beta(1, 3, size=5000)


def test_rule_of_thumb_bandwidths():
    sd = np.std(X, ddof=1)
    iqr = np.subtract(*np.quantile(X, [0.75, 0.25])) / 1.349
    s = min(sd, iqr)
    assert bandwidth_nrd0(X) == pytest.approx(0.9 * s * len(X) ** -0.2)
    assert bandwidth_nrd(X) == pytest.approx(1.06 * s * len(X) ** -0.2)


def test_dpi_bandwidth_is_close_to_normal_reference_for_normal_data():
    # for Gaussian data the plug-in and normal-reference bandwidths agree
    assert bandwidth_dpi(X) == pytest.approx(bandwidth_nrd(X), rel=0.15)


def test_dpi_falls_back_with_warning():
    with pytest.warns(UserWarning, match="falling back"):
        bw = bandwidth_dpi(np.ones(50))
    assert bw > 0


def test_unbounded_density_integrates_to_about_one():
    d = density_unbounded(X, trim=False)
    assert np.trapezoid(d.pdf, d.x) == pytest.approx(1, abs=0.01)
    assert d.cdf[0] == pytest.approx(0, abs=0.01)
    assert d.cdf[-1] == pytest.approx(1, abs=0.01)


def test_trim_limits_grid_to_data_range():
    d = density_unbounded(X, n=101)
    assert d.x[0] == X.min()
    assert d.x[-1] == X.max()
    assert len(d.x) == 101


def test_bounded_density_is_better_at_the_boundary():
    true = stats.beta(1, 3).pdf(0)  # 3
    b = density_bounded(BETA, bounds=(0, 1))
    u = density_unbounded(BETA)
    assert abs(b.pdf[0] - true) < abs(u.pdf[0] - true)
    assert b.pdf[0] == pytest.approx(true, rel=0.1)
    assert np.trapezoid(b.pdf, b.x) == pytest.approx(1, abs=0.01)


def test_estimated_bounds_contain_the_data():
    for bounder in (bounder_cdf, bounder_cooke):
        lo, hi = bounder(BETA)
        assert lo <= BETA.min() and hi >= BETA.max()
    assert bounder_range(BETA) == (BETA.min(), BETA.max())
    # symmetric data should yield symmetric bounds under bounder_cdf
    sym = np.linspace(-3, 3, 101)
    lo, hi = bounder_cdf(sym)
    assert lo == pytest.approx(-hi)
    assert lo < sym.min() - 0.1


def test_weights():
    x = np.r_[np.zeros(10), np.ones(10)] + rng.normal(0, 0.01, 20)
    w = np.r_[np.full(10, 1.0), np.full(10, 9.0)]
    d = density_unbounded(x, weights=w, bandwidth=0.1)
    near_one = d.pdf[np.argmin(np.abs(d.x - 1))]
    near_zero = d.pdf[np.argmin(np.abs(d.x - 0))]
    assert near_one / near_zero == pytest.approx(9, rel=0.1)


def test_histogram_outline():
    d = density_histogram(X, breaks=10)
    assert d.kind == "histogram"
    assert d.cdf[-1] == pytest.approx(1)
    assert d.pdf[0] == 0 and d.pdf[-1] == 0
    bars = density_histogram(X, breaks=10, outline_bars=True)
    # each bar starts and ends at zero
    assert np.sum(bars.pdf == 0) == 2 * 10


def test_weighted_quantile_and_ecdf():
    np.testing.assert_allclose(weighted_quantile(X, [0.1, 0.5]), np.quantile(X, [0.1, 0.5]))
    ecdf = weighted_ecdf([1, 2, 3], weights=[1, 1, 2])
    np.testing.assert_allclose(ecdf([0, 1, 2.5, 3]), [0, 0.25, 0.5, 1])


def test_no_warnings_for_ordinary_data():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        density_bounded(X)
