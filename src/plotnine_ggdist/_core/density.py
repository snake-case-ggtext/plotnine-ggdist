"""
Density estimators and bandwidth selectors

Independent implementations of the estimators ggdist offers for sample
data: a Gaussian KDE (`density_unbounded`), a Gaussian KDE with boundary
reflection (`density_bounded`, ggdist's default) and a histogram
(`density_histogram`). They follow the documented behaviour of their
ggdist namesakes; see the README for known differences.

This module does not import plotnine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from .kernels import KERNELS, Kernel, get_kernel

__all__ = (
    "Density",
    "bandwidth_nrd0",
    "bandwidth_nrd",
    "bandwidth_dpi",
    "bounder_range",
    "bounder_cooke",
    "bounder_cdf",
    "density_unbounded",
    "density_bounded",
    "density_histogram",
    "weighted_quantile",
    "weighted_ecdf",
    "get_density",
)

_SQRT_2PI = np.sqrt(2 * np.pi)


@dataclass
class Density:
    """
    A density estimate evaluated on a grid

    Attributes
    ----------
    x :
        Grid of values.
    pdf :
        Estimated density at `x`.
    cdf :
        Estimated cumulative distribution at `x`.
    bandwidth :
        Bandwidth used (NaN for histograms).
    n :
        Number of observations (sum of weights if weighted).
    kind :
        ``"kde"`` or ``"histogram"``.
    """

    x: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    bandwidth: float = float("nan")
    n: float = 0
    kind: str = "kde"
    extra: dict[str, Any] = field(default_factory=dict)


# Weights ----------------------------------------------------------------------


def _prep(x: Any, weights: Any | None) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.asarray(x, dtype=float).ravel()
    if weights is None:
        keep = np.isfinite(x)
        return x[keep], None
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != x.shape:
        raise ValueError("`weights` must have the same length as `x`")
    keep = np.isfinite(x) & np.isfinite(w)
    x, w = x[keep], w[keep]
    if np.any(w < 0):
        raise ValueError("`weights` must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("`weights` must not all be zero")
    return x, w / total


def weighted_quantile(
    x: Any, probs: Any, weights: Any | None = None
) -> np.ndarray:
    """
    Sample quantiles, optionally weighted

    Without weights this is ``numpy.quantile`` with its default (linear)
    method, which is R's type 7, the `quantile()` default. With weights, the
    weighted ECDF is interpolated at the midpoints of each observation's
    probability mass (equivalent to R's type 5 for equal weights).
    """
    probs = np.asarray(probs, dtype=float)
    x, w = _prep(x, weights)
    if len(x) == 0:
        return np.full(probs.shape, np.nan)
    if w is None:
        return np.quantile(x, probs)
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    p = np.cumsum(ws) - ws / 2
    return np.interp(probs, p, xs)


def weighted_ecdf(x: Any, weights: Any | None = None) -> Callable[[Any], np.ndarray]:
    """
    Return the (weighted) empirical CDF of `x` as a function
    """
    x, w = _prep(x, weights)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ws = np.full(len(xs), 1 / len(xs)) if w is None else w[order]
    cw = np.cumsum(ws)

    def ecdf(q: Any) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        idx = np.searchsorted(xs, q, side="right")
        out = np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0)
        return out

    return ecdf


# Bandwidths -------------------------------------------------------------------


def _scale_estimate(x: np.ndarray) -> float:
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    q75, q25 = np.quantile(x, [0.75, 0.25])
    iqr = (q75 - q25) / 1.349
    lo = min(sd, iqr) if iqr > 0 else sd
    if lo <= 0:
        lo = abs(x[0]) if len(x) and x[0] != 0 else 1.0
    return float(lo)


def bandwidth_nrd0(x: Any, weights: Any | None = None) -> float:
    """
    Silverman's rule of thumb (R's ``bw.nrd0``)

    ``0.9 * min(sd, IQR / 1.34) * n ** -0.2``
    """
    x, _ = _prep(x, None)
    if len(x) < 2:
        raise ValueError("need at least 2 data points")
    return 0.9 * _scale_estimate(x) * len(x) ** -0.2


def bandwidth_nrd(x: Any, weights: Any | None = None) -> float:
    """
    Scott's variation of the normal reference rule (R's ``bw.nrd``)

    ``1.06 * min(sd, IQR / 1.34) * n ** -0.2``
    """
    x, _ = _prep(x, None)
    if len(x) < 2:
        raise ValueError("need at least 2 data points")
    return 1.06 * _scale_estimate(x) * len(x) ** -0.2


def _binned_counts(x: np.ndarray, nb: int) -> tuple[np.ndarray, float]:
    lo, hi = x.min(), x.max()
    delta = (hi - lo) / (nb - 1)
    counts = np.bincount(
        np.clip(np.round((x - lo) / delta).astype(int), 0, nb - 1),
        minlength=nb,
    ).astype(float)
    return counts, delta


def _psi(counts: np.ndarray, delta: float, n: int, g: float, r: int) -> float:
    """
    Binned estimate of the density functional psi_r using a Gaussian kernel
    of bandwidth g (Wand & Jones 1995, ch. 3)
    """
    nb = len(counts)
    k = np.arange(nb) * delta / g
    phi = np.exp(-0.5 * k**2) / _SQRT_2PI
    if r == 4:
        deriv = (k**4 - 6 * k**2 + 3) * phi
    elif r == 6:
        deriv = (k**6 - 15 * k**4 + 45 * k**2 - 15) * phi
    else:  # pragma: no cover
        raise ValueError(r)
    # sum_i sum_j c_i c_j K((i - j) delta / g), using symmetry in |i - j|
    lag_sums = np.correlate(counts, counts, mode="full")[nb - 1 :]
    total = deriv[0] * lag_sums[0] + 2 * np.sum(deriv[1:] * lag_sums[1:])
    return total / (n**2 * g ** (r + 1))


def bandwidth_dpi(x: Any, weights: Any | None = None, nb: int = 1000) -> float:
    """
    Sheather-Jones direct plug-in bandwidth (two-stage)

    The default bandwidth in ggdist (there: ``bw.SJ(method = "dpi")``).
    Implemented from the two-stage plug-in rule in Wand & Jones (1995),
    using binned estimates of the density functionals. Falls back to
    [](`~plotnine_ggdist.bandwidth_nrd0`) with a warning when the estimate
    fails (e.g. many duplicated values).
    """
    import warnings

    x, _ = _prep(x, None)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 data points")
    try:
        if np.ptp(x) == 0:
            raise FloatingPointError("all values identical")
        scale = _scale_estimate(x)
        counts, delta = _binned_counts(x, nb)
        # normal-reference estimate of psi_8
        psi8 = 105 / (32 * np.sqrt(np.pi) * scale**9)
        g1 = (30 / (_SQRT_2PI * psi8 * n)) ** (1 / 9)
        psi6 = _psi(counts, delta, n, g1, 6)
        if not psi6 < 0:
            raise FloatingPointError("psi6 estimate is not negative")
        g2 = (-6 / (_SQRT_2PI * psi6 * n)) ** (1 / 7)
        psi4 = _psi(counts, delta, n, g2, 4)
        if not psi4 > 0:
            raise FloatingPointError("psi4 estimate is not positive")
        h = (1 / (2 * np.sqrt(np.pi) * psi4 * n)) ** (1 / 5)
        if not np.isfinite(h) or h <= 0:
            raise FloatingPointError("non-finite bandwidth")
        return float(h)
    except FloatingPointError as e:
        warnings.warn(
            f"bandwidth_dpi() failed ({e}); falling back to bandwidth_nrd0(). "
            "With many duplicated or discrete values, consider a histogram "
            "(density='histogram') instead of a KDE.",
            stacklevel=2,
        )
        return bandwidth_nrd0(x)


_BANDWIDTHS: dict[str, Callable[..., float]] = {
    "nrd0": bandwidth_nrd0,
    "nrd": bandwidth_nrd,
    "dpi": bandwidth_dpi,
}


def _resolve_bandwidth(bandwidth: Any, x: np.ndarray, weights: Any, adjust: float) -> float:
    if isinstance(bandwidth, str):
        try:
            fn = _BANDWIDTHS[bandwidth.removeprefix("bandwidth_")]
        except KeyError as e:
            raise ValueError(
                f"Unknown bandwidth {bandwidth!r}; use one of {sorted(_BANDWIDTHS)}, "
                "a number or a function"
            ) from e
        bw = fn(x, weights)
    elif callable(bandwidth):
        bw = float(bandwidth(x))
    else:
        bw = float(bandwidth)
    return bw * adjust


# Bounders ---------------------------------------------------------------------


def bounder_range(x: Any) -> tuple[float, float]:
    """
    Bounds of a sample: its min and max
    """
    x, _ = _prep(x, None)
    return float(x.min()), float(x.max())


def bounder_cooke(x: Any) -> tuple[float, float]:
    """
    Bounds estimated with Cooke's (1979) method (method 2.3 in Loh 1984)
    """
    x, _ = _prep(x, None)
    xs = np.sort(x)
    n = len(xs)
    i = np.arange(1, n + 1)
    w_lo = (1 - (i - 1) / n) ** n - (1 - i / n) ** n
    w_hi = (1 - (n - i) / n) ** n - (1 - (n + 1 - i) / n) ** n
    return float(2 * xs[0] - np.sum(w_lo * xs)), float(2 * xs[-1] - np.sum(w_hi * xs))


def bounder_cdf(x: Any, p: float = 0.01) -> tuple[float, float]:
    """
    Bounds estimated from the distribution of the sample's order statistics

    Treating the sample as the distribution, estimates where its minimum
    and maximum order statistics would fall (at percentile `p` of their
    distributions) and moves each bound outwards from the sample min / max
    by that distance. ``p=1`` gives the range of `x`.
    """
    x, _ = _prep(x, None)
    n = len(x)
    lo_hat, hi_hat = np.quantile(x, [1 - p ** (1 / n), p ** (1 / n)])
    return float(2 * x.min() - lo_hat), float(2 * x.max() - hi_hat)


_BOUNDERS: dict[str, Callable[[Any], tuple[float, float]]] = {
    "range": bounder_range,
    "cooke": bounder_cooke,
    "cdf": bounder_cdf,
}


# KDEs -------------------------------------------------------------------------


def _kde_sums(
    grid: np.ndarray, centers: np.ndarray, w: np.ndarray, h: float, kernel: Kernel = KERNELS["gaussian"]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted kernel pdf and cdf sums, chunked to bound memory
    """
    pdf = np.zeros(len(grid))
    cdf = np.zeros(len(grid))
    chunk = max(1, 2_000_000 // max(len(grid), 1))
    for s in range(0, len(centers), chunk):
        c = centers[s : s + chunk]
        ww = w[s : s + chunk]
        z = (grid[:, None] - c[None, :]) / h
        pdf += (kernel.pdf(z) @ ww) / h
        cdf += kernel.cdf(z) @ ww
    return pdf, cdf


_BINNED_THRESHOLD = 200_000  # len(grid) * len(x) above which to bin
_N_BINS = 4096


def _kde_binned(
    grid: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    h: float,
    bounds: tuple[float, float] | None = None,
    kernel: Kernel = KERNELS["gaussian"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    KDE (pdf, cdf) at `grid` by linear binning and FFT convolution

    With `bounds`, mass past each bound is reflected back inside it and
    the cdf starts at the lower bound. Accurate to well under 0.1% of the
    peak density when the bin width is small relative to `h`.
    """
    lo_d, hi_d = float(min(x.min(), grid.min())), float(max(x.max(), grid.max()))
    if bounds is not None:
        lo_d, hi_d = min(lo_d, bounds[0]), max(hi_d, bounds[1])
    pad = kernel.reach() * h
    a, b = lo_d - pad - (hi_d - lo_d), hi_d + pad + (hi_d - lo_d)
    if bounds is None:
        a, b = lo_d - pad, hi_d + pad
    m = _N_BINS
    delta = (b - a) / (m - 1)
    # linear binning
    pos = (x - a) / delta
    i0 = np.clip(np.floor(pos).astype(int), 0, m - 2)
    frac = pos - i0
    counts = np.bincount(i0, weights=w * (1 - frac), minlength=m) + np.bincount(
        i0 + 1, weights=w * frac, minlength=m
    )
    # kernel on the bin grid, convolved via FFT (zero padded)
    k_half = min(m - 1, int(np.ceil(kernel.reach() * h / delta)))
    kx = np.arange(-k_half, k_half + 1) * delta
    if np.isfinite(kernel.support):
        # exact mass of each bin (accurate for kernels with corners)
        edges = np.r_[kx - delta / 2, kx[-1] + delta / 2] / h
        kern = np.diff(kernel.cdf(edges)) / delta
    else:
        kern = kernel.pdf(kx / h) / h
    size = int(2 ** np.ceil(np.log2(m + len(kern))))
    dens = np.fft.irfft(np.fft.rfft(counts, size) * np.fft.rfft(kern, size), size)
    dens = dens[k_half : k_half + m]
    xs = a + np.arange(m) * delta

    if bounds is None:
        pdf = np.interp(grid, xs, dens)
        cum = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1]) / 2 * delta)])
        cdf = np.interp(grid, xs, cum)
        return np.maximum(pdf, 0), np.clip(cdf / cum[-1], 0, 1)

    lo_b, hi_b = bounds

    def reflected(v: np.ndarray) -> np.ndarray:
        return (
            np.interp(v, xs, dens)
            + np.interp(2 * lo_b - v, xs, dens)
            + np.interp(2 * hi_b - v, xs, dens)
        )

    fine = np.linspace(lo_b, hi_b, m)
    fd = np.maximum(reflected(fine), 0)
    cum = np.concatenate([[0], np.cumsum((fd[1:] + fd[:-1]) / 2 * (fine[1] - fine[0]))])
    total = cum[-1] if cum[-1] > 0 else 1.0
    pdf = np.maximum(reflected(grid), 0) / total
    cdf = np.interp(grid, fine, cum) / total
    return pdf, np.clip(cdf, 0, 1)


def density_unbounded(
    x: Any,
    weights: Any | None = None,
    n: int = 501,
    bandwidth: Any = "dpi",
    adjust: float = 1,
    trim: bool = True,
    cut: float = 3,
    kernel: str = "gaussian",
) -> Density:
    """
    Kernel density estimate

    Parameters
    ----------
    x :
        Sample.
    weights :
        Optional non-negative weights, one per value of `x`.
    n :
        Number of grid points.
    bandwidth :
        A number (the kernel's standard deviation), a function of the
        sample, or one of ``"dpi"`` (default), ``"nrd0"``, ``"nrd"``.
    adjust :
        Multiplier for the bandwidth.
    trim :
        If `True`, evaluate only over the range of the data; otherwise
        extend the grid by `cut` bandwidths on each side.
    kernel :
        ``"gaussian"`` (default), ``"rectangular"``, ``"triangular"``,
        ``"epanechnikov"``, ``"biweight"``, ``"cosine"`` or
        ``"optcosine"``; partial names work. Bandwidths are the kernel's
        standard deviation, as in R's ``density()``.
    """
    k = get_kernel(kernel)
    x, w = _prep(x, weights)
    if len(x) == 0:
        raise ValueError("no finite values")
    ww = np.full(len(x), 1 / len(x)) if w is None else w
    h = _resolve_bandwidth(bandwidth, x, weights, adjust) if len(x) > 1 else 1.0
    lo, hi = x.min(), x.max()
    if not trim:
        reach = min(cut, k.support)
        lo, hi = lo - reach * h, hi + reach * h
    grid = np.linspace(lo, hi, n)
    if len(grid) * len(x) > _BINNED_THRESHOLD:
        pdf, cdf = _kde_binned(grid, x, ww, h, kernel=k)
    else:
        pdf, cdf = _kde_sums(grid, x, ww, h, k)
    return Density(grid, pdf, cdf, bandwidth=h, n=len(x), kind="kde")


def density_bounded(
    x: Any,
    weights: Any | None = None,
    n: int = 501,
    bandwidth: Any = "dpi",
    adjust: float = 1,
    trim: bool = True,
    bounds: tuple[float | None, float | None] = (None, None),
    bounder: str | Callable[[Any], tuple[float, float]] = "cdf",
    kernel: str = "gaussian",
) -> Density:
    """
    Kernel density estimate with boundary reflection

    The default density estimator for sample data, as in ggdist. Missing
    `bounds` are estimated with `bounder` (``"cdf"``, ``"cooke"`` or
    ``"range"``, or a function returning ``(lower, upper)``); a bound is
    only used when it is at or beyond the range of the data. Mass that a
    kernel places past a bound is reflected back inside it.

    See [](`~plotnine_ggdist.density_unbounded`) for the other parameters.
    """
    k = get_kernel(kernel)
    x, w = _prep(x, weights)
    if len(x) == 0:
        raise ValueError("no finite values")
    ww = np.full(len(x), 1 / len(x)) if w is None else w
    h = _resolve_bandwidth(bandwidth, x, weights, adjust) if len(x) > 1 else 1.0

    lo_b, hi_b = (None if (b is None or np.isnan(b)) else float(b) for b in bounds)
    if lo_b is None or hi_b is None:
        fn = _BOUNDERS[bounder] if isinstance(bounder, str) else bounder
        est_lo, est_hi = fn(x) if len(x) > 1 else (x[0], x[0])
        lo_b = est_lo if lo_b is None else lo_b
        hi_b = est_hi if hi_b is None else hi_b
    lo_b = min(lo_b, x.min())
    hi_b = max(hi_b, x.max())

    grid = np.linspace(x.min(), x.max(), n) if trim else np.linspace(lo_b, hi_b, n)
    if len(grid) * len(x) > _BINNED_THRESHOLD:
        pdf, cdf = _kde_binned(grid, x, ww, h, bounds=(lo_b, hi_b), kernel=k)
        return Density(grid, pdf, cdf, bandwidth=h, n=len(x), kind="kde", extra={"bounds": (lo_b, hi_b)})
    p0, c0 = _kde_sums(grid, x, ww, h, k)
    p1, c1 = _kde_sums(grid, 2 * lo_b - x, ww, h, k)
    p2, c2 = _kde_sums(grid, 2 * hi_b - x, ww, h, k)
    # cdf from lo_b: integrate each kernel component from the lower bound
    at_lo = _kde_sums(np.array([lo_b]), x, ww, h, k)[1] + _kde_sums(
        np.array([lo_b]), 2 * lo_b - x, ww, h
    )[1] + _kde_sums(np.array([lo_b]), 2 * hi_b - x, ww, h, k)[1]
    pdf = p0 + p1 + p2
    cdf = (c0 + c1 + c2) - at_lo
    at_hi = (
        _kde_sums(np.array([hi_b]), x, ww, h, k)[1]
        + _kde_sums(np.array([hi_b]), 2 * lo_b - x, ww, h, k)[1]
        + _kde_sums(np.array([hi_b]), 2 * hi_b - x, ww, h, k)[1]
        - at_lo
    )
    total = float(at_hi[0]) if at_hi[0] > 0 else 1.0
    return Density(
        grid,
        pdf / total,
        np.clip(cdf / total, 0, 1),
        bandwidth=h,
        n=len(x),
        kind="kde",
        extra={"bounds": (lo_b, hi_b)},
    )


# Histogram --------------------------------------------------------------------


def density_histogram(
    x: Any,
    weights: Any | None = None,
    breaks: int | str | np.ndarray = "scott",
    outline_bars: bool = False,
) -> Density:
    """
    Histogram density estimator

    Parameters
    ----------
    breaks :
        Number of bins, bin edges, or a rule understood by
        ``numpy.histogram_bin_edges`` (``"scott"`` (default), ``"sturges"``,
        ``"fd"``, ...).
    outline_bars :
        If `True`, the returned outline drops to zero between bars so that
        each bar is drawn with its own outline.

    Notes
    -----
    The returned `x`/`pdf` trace the outline of the bars (including each bin's
    midpoint), so it can be drawn directly as a slab.
    """
    x, w = _prep(x, weights)
    if len(x) == 0:
        raise ValueError("no finite values")
    if isinstance(breaks, str):
        breaks = breaks.lower()
    if np.ptp(x) == 0:
        edges = np.array([x[0] - 0.5, x[0] + 0.5])
    else:
        edges = np.histogram_bin_edges(x, bins=breaks)
    dens, edges = np.histogram(x, bins=edges, weights=w, density=True)
    widths = np.diff(edges)
    cum = np.concatenate([[0], np.cumsum(dens * widths)])

    xs, ps, cs = [], [], []
    for i, d in enumerate(dens):
        left, right = edges[i], edges[i + 1]
        mid = 0.5 * (left + right)
        cmid = 0.5 * (cum[i] + cum[i + 1])
        if outline_bars or i == 0:
            xs.append(left)
            ps.append(0.0)
            cs.append(cum[i])
        xs += [left, mid, right]
        ps += [d, d, d]
        cs += [cum[i], cmid, cum[i + 1]]
        if outline_bars or i == len(dens) - 1:
            xs.append(right)
            ps.append(0.0)
            cs.append(cum[i + 1])
    return Density(
        np.asarray(xs),
        np.asarray(ps),
        np.asarray(cs),
        n=len(x),
        kind="histogram",
        extra={"breaks": edges},
    )


# Lookup -----------------------------------------------------------------------

DensityName = Literal["bounded", "unbounded", "histogram"]

_DENSITIES: dict[str, Callable[..., Density]] = {
    "bounded": density_bounded,
    "unbounded": density_unbounded,
    "histogram": density_histogram,
}


def get_density(density: str | Callable[..., Density]) -> Callable[..., Density]:
    """
    Resolve a density estimator from a name or callable
    """
    if callable(density):
        return density
    try:
        return _DENSITIES[density.removeprefix("density_")]
    except KeyError as e:
        raise ValueError(
            f"Unknown density {density!r}; use one of {sorted(_DENSITIES)} or a function"
        ) from e
