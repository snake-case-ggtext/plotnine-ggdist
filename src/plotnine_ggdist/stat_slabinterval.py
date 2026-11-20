"""
stat_slabinterval and its shortcut stats

One stat turns samples (``x``/``y``) or frozen scipy.stats distributions
(``xdist``/``ydist``) into the two kinds of rows that
[](`~plotnine_ggdist.geom_slabinterval`) draws:

- ``datatype == "slab"``: a grid of values with the ``pdf``, ``cdf``,
  ``n``, ``.width`` and ``level`` at each value;
- ``datatype == "interval"``: one row per interval with the point summary
  (``x``/``y``), the interval ends (``xmin``/``xmax`` or ``ymin``/``ymax``),
  ``.width`` and ``level``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from plotnine.exceptions import PlotnineError
from plotnine.mapping.evaluation import after_stat
from plotnine.stats.stat import stat

from ._core.density import Density, get_density, weighted_ecdf
from ._core.dist import dist_cdf, dist_limits, dist_pdf, is_discrete, is_dist
from ._core.intervals import parse_point_interval, summarise
from ._orientation import HORIZONTAL, VERTICAL, flip, normalize_orientation
from ._utils import compute_panel_by_group

if TYPE_CHECKING:
    from plotnine.facets.layout import Layout
    from plotnine.iapi import pos_scales

__all__ = (
    "stat_slabinterval",
    "stat_halfeye",
    "stat_eye",
    "stat_slab",
    "stat_ccdfinterval",
    "stat_cdfinterval",
    "stat_gradientinterval",
    "stat_histinterval",
    "stat_pointinterval",
    "stat_interval",
)

SLAB_COLUMNS = ("x", "pdf", "cdf", "n", ".width")
INTERVAL_COLUMNS = ("x", "xmin", "xmax", ".width")


def width_levels(widths: Any) -> list[float]:
    """
    Levels of the `level` variable: widths, widest first

    Widest first so that sequential palettes put the darkest colour on
    the narrowest interval, as in ggdist.
    """
    return sorted({float(w) for w in np.atleast_1d(widths)}, reverse=True)


def as_level(values: Any, levels: list[float]) -> pd.Categorical:
    return pd.Categorical(values, categories=levels, ordered=True)


class stat_slabinterval(stat):
    """
    Slab (density, CDF, ...) and interval summaries of distributions

    The general stat behind [](`~plotnine_ggdist.stat_halfeye`),
    [](`~plotnine_ggdist.stat_eye`), [](`~plotnine_ggdist.stat_slab`),
    [](`~plotnine_ggdist.stat_pointinterval`) and the other shortcuts.
    Port of `ggdist::stat_slabinterval()`.

    Map sample data to ``x`` (or ``y``), or a column of frozen
    ``scipy.stats`` distributions to ``xdist`` (or ``ydist``). The other
    position aesthetic identifies groups.

    Parameters
    ----------
    mapping, data, geom, position, na_rm, inherit_aes, show_legend :
        As for any plotnine stat.
    orientation :
        ``"horizontal"`` (alias ``"y"``): values along x, groups along y.
        ``"vertical"`` (alias ``"x"``): values along y, groups along x.
        ``None`` (default) detects it: ``xdist`` or a discrete y means
        horizontal, ``ydist`` or a discrete x means vertical; with only
        ``x`` mapped it is horizontal, with only ``y`` vertical; with two
        continuous positions it is vertical, as in plotnine's own geoms.
    point_interval :
        Point summary and interval function: ``"median_qi"`` (default),
        ``"mean_hdi"``, ``"mode_hdci"``, ... or a ``(point, interval)``
        tuple of names or functions (see
        [](`~plotnine_ggdist.point_interval`)).
    width :
        Probability mass of the intervals, ggdist's ``.width``. Default
        ``(0.66, 0.95)``.
    density :
        Density estimator for samples: ``"bounded"`` (default),
        ``"unbounded"``, ``"histogram"`` or a function
        ``f(x, weights=..., n=...)`` returning a
        [](`~plotnine_ggdist.Density`).
    bandwidth, adjust, trim :
        Passed to the KDE: bandwidth (default ``"dpi"``), bandwidth
        multiplier (default 1) and whether to trim the estimate to the range
        of the data (default `True`).
    breaks, outline_bars :
        Passed to the histogram estimator.
    p_limits :
        ``(lower, upper)`` probabilities limiting the slab of an analytical
        distribution. ``None`` uses the end of the support if finite,
        otherwise the 0.001 / 0.999 quantile.
    limits :
        ``(lower, upper)`` values the slab never extends past; ``None``
        leaves a side alone.
    n :
        Number of points the slab function is evaluated at. Default 501.
    expand :
        For samples, extend the slab to the limits of the scale (useful for
        CDF/CCDF slabs). A bool, or a pair for (lower, upper).

    Computed variables
    ------------------
    datatype :
        ``"slab"`` or ``"interval"``.
    x or y :
        Slab: the input to the slab function. Interval: the point summary.
    xmin, xmax or ymin, ymax :
        Interval ends.
    pdf, cdf :
        Slab: density and cumulative distribution.
    n :
        Slab: number of draws (``inf`` for analytical distributions).
    .width :
        Interval: its probability mass. Slab: that of the narrowest
        interval containing the value (NaN if none).
    level :
        ``.width`` as an ordered categorical, widest first. Use this name in
        [](`~plotnine.after_stat`) (``.width`` is not a Python identifier).
    """

    REQUIRED_AES: set[str] = set()
    DEFAULT_AES = {"thickness": after_stat("pdf"), "weight": None}
    NON_MISSING_AES: set[str] = set()
    CREATES = {"datatype", "pdf", "cdf", "n", ".width", "level", "thickness", "orientation"}
    DEFAULT_PARAMS = {
        "geom": "slabinterval",
        "position": "identity",
        "na_rm": False,
        "orientation": None,
        "point_interval": "median_qi",
        "width": (0.66, 0.95),
        "density": "bounded",
        "bandwidth": "dpi",
        "adjust": 1,
        "trim": True,
        "breaks": "scott",
        "outline_bars": False,
        "p_limits": (None, None),
        "limits": None,
        "n": 501,
        "expand": False,
        # which sub-geometries this stat computes
        "show_slab": True,
        "show_interval": True,
        # geom parameters a shortcut may need to set (copied to the geom)
        "normalize": "all",
        "fill_type": "segments",
    }

    # Orientation detection -----------------------------------------------

    def compute_layer(self, data: pd.DataFrame, layout: Layout) -> pd.DataFrame:
        orientation = normalize_orientation(self.params.get("orientation"))
        if orientation is None:
            orientation = self._detect_orientation(data, layout)
        self._orientation = orientation
        data = flip(data, orientation)
        if "x" not in data.columns and "xdist" not in data.columns:
            raise PlotnineError(
                f"{type(self).__name__} needs sample values (x or y) or "
                "distributions (xdist or ydist)"
            )
        if "y" not in data.columns:
            data["y"] = 0.0
        # the range of all distributions in the layer, for `expand`
        self._dist_range = None
        if "xdist" in data.columns:
            lims = [dist_limits(d, self.params["p_limits"]) for d in data["xdist"] if is_dist(d)]
            if lims:
                self._dist_range = (min(l for l, _ in lims), max(h for _, h in lims))
        result = super().compute_layer(data, layout)
        if len(result):
            result["orientation"] = orientation
        return flip(result, orientation)

    def _detect_orientation(self, data: pd.DataFrame, layout: Layout) -> str:
        cols = set(data.columns)
        if "xdist" in cols:
            return HORIZONTAL
        if "ydist" in cols:
            return VERTICAL
        has_x, has_y = "x" in cols, "y" in cols
        if has_x and not has_y:
            return HORIZONTAL
        if has_y and not has_x:
            return VERTICAL
        sx = layout.panel_scales_x[0] if layout.panel_scales_x else None
        sy = layout.panel_scales_y[0] if layout.panel_scales_y else None
        if _is_discrete(sy) and not _is_discrete(sx):
            return HORIZONTAL
        return VERTICAL

    def compute_panel(self, data: pd.DataFrame, scales: pos_scales) -> pd.DataFrame:
        if self._orientation == VERTICAL:
            scales = SimpleNamespace(x=scales.y, y=scales.x)
        return compute_panel_by_group(data, scales, self.compute_group)

    # Computation ----------------------------------------------------------

    def compute_group(self, data: pd.DataFrame, scales: pos_scales) -> pd.DataFrame:
        params = self.params
        point_fn, interval_fn, _, _ = parse_point_interval(params["point_interval"])
        levels = width_levels(params["width"])
        trans = _get_trans(scales.x)

        if "xdist" in data.columns:
            pieces = []
            for _, row in data.iterrows():
                d = row["xdist"]
                if d is None or (isinstance(d, float) and np.isnan(d)):
                    continue
                if not is_dist(d):
                    raise PlotnineError(
                        "xdist/ydist must hold frozen scipy.stats distributions, "
                        f"got {type(d).__name__}"
                    )
                out = self._compute_one(d, None, point_fn, interval_fn, levels, trans, scales)
                # carry the row's other aesthetics (several dists per group)
                extra = row.drop(labels=[c for c in row.index if c in out.columns or c == "xdist"])
                for k, v in extra.items():
                    out[k] = v
                pieces.append(out)
            if not pieces:
                return pd.DataFrame()
            out = pd.concat(pieces, ignore_index=True)
        else:
            # one distribution per position on the group axis (a continuous
            # position axis can hold several within a group)
            pieces = []
            has_w = _has_values(data, "weight")
            for y, sub in data.groupby("y", sort=True):
                x = sub["x"].to_numpy(dtype=float)
                w = sub["weight"].to_numpy(dtype=float) if has_w else None
                piece = self._compute_one(x, w, point_fn, interval_fn, levels, None, scales)
                piece["y"] = y
                pieces.append(piece)
            if not pieces:
                return pd.DataFrame()
            out = pd.concat(pieces, ignore_index=True)
        out["level"] = as_level(out[".width"], levels)
        return out

    def _compute_one(
        self,
        x: Any,
        weights: np.ndarray | None,
        point_fn: Any,
        interval_fn: Any,
        levels: list[float],
        trans: Any,
        scales: Any,
    ) -> pd.DataFrame:
        params = self.params
        parts = []
        intervals = None

        if params["show_interval"] or params["show_slab"]:
            # intervals on the original scale (for dists), transformed below
            intervals = summarise(x, params["width"], point_fn, interval_fn, weights=weights)

        if params["show_slab"]:
            slab = self._slab(x, weights, intervals, trans, scales)
            parts.append(slab)

        if params["show_interval"]:
            assert intervals is not None
            iv = pd.DataFrame(
                {
                    "datatype": "interval",
                    "x": intervals["y"],
                    "xmin": intervals["ymin"],
                    "xmax": intervals["ymax"],
                    ".width": intervals[".width"],
                }
            )
            if trans is not None:
                for c in ("x", "xmin", "xmax"):
                    iv[c] = _transform(trans, iv[c].to_numpy(dtype=float))
            parts.append(iv)

        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _slab(
        self,
        x: Any,
        weights: np.ndarray | None,
        intervals: pd.DataFrame | None,
        trans: Any,
        scales: Any,
    ) -> pd.DataFrame:
        params = self.params
        n_grid = int(params["n"])

        if is_dist(x):
            lo, hi = dist_limits(x, params["p_limits"])
            dist_range = getattr(self, "_dist_range", None)
            if dist_range is not None and not is_discrete(x):
                sup_lo, sup_hi = x.support()
                lo = max(min(lo, dist_range[0]), float(sup_lo))
                hi = min(max(hi, dist_range[1]), float(sup_hi))
            lo, hi = _expand_limits(lo, hi, params["expand"], scales, trans, dist_range)
            lo, hi = _apply_limits(lo, hi, params["limits"])
            if trans is not None and not is_discrete(x):
                # keep both ends finite on the transformed scale (e.g. 0 on log)
                t_lo, t_hi = _transform(trans, np.array([lo, hi]))
                if not np.isfinite(t_lo):
                    lo = float(x.ppf(0.001))
                if not np.isfinite(t_hi):
                    hi = float(x.ppf(0.999))
            if is_discrete(x):
                k = np.arange(np.ceil(lo), np.floor(hi) + 1)
                pmf = dist_pdf(x, k)
                cdf = dist_cdf(x, k)
                # step outline: each integer gets a bar of width 1
                xs = np.repeat(k, 2) + np.tile([-0.5, 0.5], len(k))
                pdf = np.repeat(pmf, 2)
                cdfs = np.repeat(cdf, 2)
                n = np.inf
            else:
                if trans is not None:
                    # evenly spaced on the transformed scale
                    t_lo, t_hi = _transform(trans, np.array([lo, hi]))
                    xs = np.asarray(trans.inverse(np.linspace(t_lo, t_hi, n_grid)), dtype=float)
                else:
                    xs = np.linspace(lo, hi, n_grid)
                if intervals is not None:
                    # put interval ends and points on the grid for crisp fills
                    extra = intervals[["y", "ymin", "ymax"]].to_numpy().ravel()
                    extra = extra[np.isfinite(extra) & (extra >= lo) & (extra <= hi)]
                    xs = np.unique(np.concatenate([xs, extra]))
                pdf = dist_pdf(x, xs)
                cdfs = dist_cdf(x, xs)
                n = np.inf
            if trans is not None:
                t = _transform(trans, xs)
                ok = np.isfinite(t)
                xs_t, pdf, cdfs = t[ok], pdf[ok], cdfs[ok]
                # density on the transformed scale: f(x) |dx/dt|
                if len(xs_t) > 2 and not is_discrete(x):
                    dxdt = np.gradient(xs[ok], xs_t)
                    pdf = pdf * np.abs(dxdt)
                xs = xs_t
        else:
            values = np.asarray(x, dtype=float)
            keep = np.isfinite(values)
            values = values[keep]
            w = None if weights is None else weights[keep]
            if len(values) == 0:
                return pd.DataFrame(columns=["datatype", *SLAB_COLUMNS])
            dens = self._density(values, w, n_grid)
            xs, pdf = dens.x, dens.pdf
            cdfs = dens.cdf if dens.kind == "histogram" else weighted_ecdf(values, w)(xs)
            n = float(len(values)) if w is None else float(np.sum(w))
            xs, pdf, cdfs = _expand(xs, pdf, cdfs, params["expand"], scales)
            lims = params["limits"]
            if lims is not None:
                lo, hi = _apply_limits(-np.inf, np.inf, lims)
                ok = (xs >= lo) & (xs <= hi)
                xs, pdf, cdfs = xs[ok], pdf[ok], cdfs[ok]

        slab = pd.DataFrame(
            {"datatype": "slab", "x": xs, "pdf": pdf, "cdf": cdfs, "n": n}
        )
        slab[".width"] = _slab_widths(slab["x"].to_numpy(), intervals, trans)
        return slab

    def _density(self, values: np.ndarray, w: np.ndarray | None, n: int) -> Density:
        params = self.params
        est = get_density(params["density"])
        name = getattr(est, "__name__", "")
        if name == "density_histogram":
            return est(values, weights=w, breaks=params["breaks"], outline_bars=params["outline_bars"])
        if name in ("density_bounded", "density_unbounded"):
            if len(np.unique(values)) < 2:
                # a single repeated value: a spike
                v = values[0]
                return Density(np.array([v, v]), np.array([0.0, np.inf]), np.array([0.0, 1.0]), n=len(values))
            return est(
                values,
                weights=w,
                n=n,
                bandwidth=params["bandwidth"],
                adjust=params["adjust"],
                trim=params["trim"],
            )
        return est(values, weights=w, n=n)


def _is_discrete(scale: Any) -> bool:
    from plotnine.scales.scale_discrete import scale_discrete

    return isinstance(scale, scale_discrete)


def _has_values(data: pd.DataFrame, col: str) -> bool:
    return col in data.columns and data[col].notna().any()


def _apply_limits(lo: float, hi: float, limits: Any) -> tuple[float, float]:
    if limits is None:
        return lo, hi
    l_lo, l_hi = limits
    if l_lo is not None and not (isinstance(l_lo, float) and np.isnan(l_lo)):
        lo = max(lo, float(l_lo))
    if l_hi is not None and not (isinstance(l_hi, float) and np.isnan(l_hi)):
        hi = min(hi, float(l_hi))
    return lo, hi


def _expand(xs, pdf, cdf, expand, scales):
    if isinstance(expand, (bool, np.bool_)):
        expand = (bool(expand), bool(expand))
    lo_e, hi_e = expand
    if not (lo_e or hi_e):
        return xs, pdf, cdf
    try:
        lo, hi = scales.x.dimension()
    except Exception:
        return xs, pdf, cdf
    if lo_e and lo < xs[0]:
        xs, pdf, cdf = np.r_[lo, xs], np.r_[0.0, pdf], np.r_[0.0, cdf]
    if hi_e and hi > xs[-1]:
        xs, pdf, cdf = np.r_[xs, hi], np.r_[pdf, 0.0], np.r_[cdf, 1.0]
    return xs, pdf, cdf


def _expand_limits(
    lo: float, hi: float, expand: Any, scales: Any, trans: Any, dist_range: Any = None
) -> tuple[float, float]:
    """
    For analytical distributions: widen (lo, hi) to the scale's limits and
    to the range of all the layer's distributions
    """
    if isinstance(expand, (bool, np.bool_)):
        expand = (bool(expand), bool(expand))
    if not any(expand):
        return lo, hi
    s_lo, s_hi = np.inf, -np.inf
    try:
        is_empty = getattr(scales.x, "is_empty", None)
        if is_empty is None or not is_empty():
            d_lo, d_hi = (float(v) for v in scales.x.dimension())
            if np.isfinite(d_lo) and np.isfinite(d_hi) and d_hi > d_lo:
                if trans is not None:
                    d_lo, d_hi = (float(v) for v in trans.inverse(np.array([d_lo, d_hi], dtype=float)))
                s_lo, s_hi = d_lo, d_hi
    except Exception:
        pass
    if dist_range is not None:
        s_lo, s_hi = min(s_lo, dist_range[0]), max(s_hi, dist_range[1])
    if expand[0] and np.isfinite(s_lo):
        lo = min(lo, s_lo)
    if expand[1] and np.isfinite(s_hi):
        hi = max(hi, s_hi)
    return lo, hi


def _slab_widths(xs: np.ndarray, intervals: pd.DataFrame | None, trans: Any) -> np.ndarray:
    """
    Width of the narrowest interval containing each slab value
    """
    out = np.full(len(xs), np.nan)
    if intervals is None or not len(intervals):
        return out
    iv = intervals.sort_values(".width")
    lo = iv["ymin"].to_numpy(dtype=float)
    hi = iv["ymax"].to_numpy(dtype=float)
    if trans is not None:
        lo, hi = _transform(trans, lo), _transform(trans, hi)
    for l, h, w in zip(lo, hi, iv[".width"].to_numpy(dtype=float), strict=True):
        inside = np.isnan(out) & (xs >= l) & (xs <= h)
        out[inside] = w
    return out


def _get_trans(scale: Any) -> Any:
    """
    The scale's transform, or None if it is the identity (or missing)
    """
    trans = getattr(scale, "_trans", None)
    if trans is None or not hasattr(trans, "transform"):
        trans = getattr(scale, "trans", None)
    if trans is None or not hasattr(trans, "transform"):
        return None
    if getattr(trans, "transform_is_linear", False) and type(trans).__name__ == "identity_trans":
        return None
    name = type(trans).__name__
    if name in ("identity_trans", "IdentityTransform") or "identity" in name.lower():
        return None
    return trans


def _transform(trans: Any, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.asarray(trans.transform(np.asarray(x, dtype=float)), dtype=float)


# Shortcut stats -----------------------------------------------------------------


def _shortcut(
    name: str,
    doc: str,
    params: dict[str, Any] | None = None,
    aes: dict[str, Any] | None = None,
    set_aes: dict[str, Any] | None = None,
    base: type[stat_slabinterval] = stat_slabinterval,
) -> type[stat_slabinterval]:
    """
    Make a shortcut stat

    `params` are stat parameters (the layer copies them to the geom, so geom
    parameters such as `normalize` can be set here too); `aes` are computed
    default mappings (``after_stat``); `set_aes` are constant aesthetics for
    the geom (e.g. ``side="both"``), applied unless the user sets them.
    """
    set_aes = dict(set_aes or {})

    def __init__(self, mapping=None, data=None, **kwargs):
        mapped = set(mapping.keys()) if mapping is not None and hasattr(mapping, "keys") else set()
        for k, v in set_aes.items():
            if k not in mapped:
                kwargs.setdefault(k, v)
        base.__init__(self, mapping, data, **kwargs)

    attrs: dict[str, Any] = {
        "__doc__": doc,
        "__init__": __init__,
        "DEFAULT_PARAMS": base.DEFAULT_PARAMS | (params or {}),
        "DEFAULT_AES": base.DEFAULT_AES | (aes or {}),
        "_SET_AES": set_aes,
    }
    return type(name, (base,), attrs)  # type: ignore[return-value]


stat_halfeye = _shortcut(
    "stat_halfeye",
    """
    Half-eye plot: density slab with point and intervals

    Port of `ggdist::stat_halfeye()`. A slab of the density on one side
    plus a point summary and intervals; see
    [](`~plotnine_ggdist.stat_slabinterval`) for parameters.
    """,
)

stat_eye = _shortcut(
    "stat_eye",
    """
    Eye (violin) plot: mirrored density slab with point and intervals

    Port of `ggdist::stat_eye()`; ``side="both"``.
    See [](`~plotnine_ggdist.stat_slabinterval`).
    """,
    set_aes={"side": "both"},
)

stat_slab = _shortcut(
    "stat_slab",
    """
    Slab (density) only

    Port of `ggdist::stat_slab()`; draws only the slab.
    See [](`~plotnine_ggdist.stat_slabinterval`).
    """,
    params={"geom": "slab", "show_interval": False},
)

stat_ccdfinterval = _shortcut(
    "stat_ccdfinterval",
    """
    CCDF bar plot: complementary CDF slab with point and intervals

    Port of `ggdist::stat_ccdfinterval()`: thickness is ``1 - cdf``, not
    normalised, the slab expands to the scale limits, and the interval is
    centred (``justification=0.5``) with the slab on top.
    See [](`~plotnine_ggdist.stat_slabinterval`).
    """,
    params={"expand": True, "normalize": "none"},
    aes={"thickness": after_stat("1 - cdf")},
    set_aes={"justification": 0.5},
)

stat_cdfinterval = _shortcut(
    "stat_cdfinterval",
    """
    CDF bar plot: CDF slab with point and intervals

    Port of `ggdist::stat_cdfinterval()`. See
    [](`~plotnine_ggdist.stat_ccdfinterval`).
    """,
    params={"expand": True, "normalize": "none"},
    aes={"thickness": after_stat("cdf")},
    set_aes={"justification": 0.5},
)

stat_gradientinterval = _shortcut(
    "stat_gradientinterval",
    """
    Gradient plot: a band whose opacity follows the density

    Port of `ggdist::stat_gradientinterval()`: the slab fills the whole
    band (``thickness=1``) and its alpha is the normalised density.
    See [](`~plotnine_ggdist.stat_slabinterval`).
    """,
    params={"fill_type": "gradient"},
    aes={"thickness": None, "slab_alpha": after_stat("pdf / pdf.max()")},
    set_aes={"thickness": 1, "justification": 0.5},
)

stat_histinterval = _shortcut(
    "stat_histinterval",
    """
    Histogram slab with point and intervals

    Port of `ggdist::stat_histinterval()`; ``density="histogram"``.
    See [](`~plotnine_ggdist.stat_slabinterval`).
    """,
    params={"density": "histogram"},
)

stat_pointinterval = _shortcut(
    "stat_pointinterval",
    """
    Point summary with multiple intervals

    Port of `ggdist::stat_pointinterval()`; no slab.
    See [](`~plotnine_ggdist.stat_slabinterval`).
    """,
    params={"geom": "pointinterval", "show_slab": False},
    aes={"thickness": None},
)

stat_interval = _shortcut(
    "stat_interval",
    """
    Multiple intervals coloured by their width

    Port of `ggdist::stat_interval()`: thick interval bars, one colour per
    ``level`` (widest first, so sequential palettes are darkest on the
    narrowest interval). Default widths ``(0.5, 0.8, 0.95)``.
    See [](`~plotnine_ggdist.stat_slabinterval`).
    """,
    params={"geom": "interval", "show_slab": False, "width": (0.5, 0.8, 0.95)},
    aes={"thickness": None, "color": after_stat("level")},
)
