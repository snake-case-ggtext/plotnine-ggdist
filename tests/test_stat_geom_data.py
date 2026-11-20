"""
Tests of what the stats compute and how the geoms lay it out, checked on
the built layer data (no image comparison)
"""

import numpy as np
import pandas as pd
import pytest
from plotnine import aes, after_stat, coord_flip, facet_wrap, ggplot, scale_x_log10
from plotnine.exceptions import PlotnineError
from scipy import stats

from plotnine_ggdist import (
    geom_interval,
    geom_lineribbon,
    geom_pointinterval,
    geom_slab,
    median_qi,
    stat_ccdfinterval,
    stat_eye,
    stat_halfeye,
    stat_interval,
    stat_lineribbon,
    stat_pointinterval,
    stat_slab,
)

rng = np.random.default_rng(42)
SAMPLES = pd.DataFrame(
    {"g": np.tile(["a", "b", "c"], 400), "x": rng.normal(np.tile([0.0, 2.0, 4.0], 400), 1)}
)
DISTS = pd.DataFrame({"g": ["a", "b"], "d": [stats.norm(0, 1), stats.gamma(3)]})


def slab(d):
    return d[d["datatype"] == "slab"]


def interval(d):
    return d[d["datatype"] == "interval"]


# stat output --------------------------------------------------------------------------


def test_halfeye_samples_rows_and_intervals(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye())
    assert set(d["datatype"]) == {"slab", "interval"}
    assert (d["orientation"] == "horizontal").all()
    iv = interval(d)
    assert len(iv) == 6  # 3 groups x 2 widths
    expected = median_qi(SAMPLES, "x", by="g", width=[0.66, 0.95])
    np.testing.assert_allclose(iv["xmin"].to_numpy(), expected[".lower"].to_numpy())
    np.testing.assert_allclose(iv["xmax"].to_numpy(), expected[".upper"].to_numpy())
    np.testing.assert_allclose(iv["x"].to_numpy(), expected["x"].to_numpy())
    assert (slab(d)["n"] == 400).all()


def test_level_is_ordered_widest_first(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye(width=[0.5, 0.8, 0.95]))
    level = interval(d)["level"]
    assert list(level.cat.categories) == [0.95, 0.8, 0.5]
    assert level.cat.ordered


def test_slab_width_is_narrowest_containing_interval(layer_data):
    d = layer_data(ggplot(DISTS.iloc[[0]], aes(y="g", xdist="d")) + stat_halfeye())
    s = slab(d)
    inner = s[(s["x"] > -0.9) & (s["x"] < 0.9)]
    outer = s[(s["x"] > 1.2) & (s["x"] < 1.9)]
    tail = s[s["x"] > 2.1]
    assert (inner[".width"] == 0.66).all()
    assert (outer[".width"] == 0.95).all()
    assert tail[".width"].isna().all()


def test_dist_slab_is_the_pdf(layer_data):
    d = layer_data(ggplot(DISTS, aes(y="g", xdist="d")) + stat_slab(normalize="none"))
    s = slab(d)
    for g, dist in zip(DISTS["g"], DISTS["d"], strict=True):
        part = s[s["group"] == {"a": 1, "b": 2}[g]]
        np.testing.assert_allclose(part["pdf"], dist.pdf(part["x"]))
        np.testing.assert_allclose(part["thickness"], dist.pdf(part["x"]))
        assert np.isinf(part["n"]).all()
    # normal: 0.001 to 0.999 quantiles; gamma: from 0 (finite support)
    a = s[s["group"] == 1]["x"]
    b = s[s["group"] == 2]["x"]
    assert a.min() == pytest.approx(stats.norm.ppf(0.001))
    assert b.min() == 0


def test_p_limits_and_limits(layer_data):
    d = layer_data(
        ggplot(DISTS.iloc[[0]], aes(y="g", xdist="d"))
        + stat_slab(p_limits=(0.05, 0.95), limits=(None, 1))
    )
    x = slab(d)["x"]
    assert x.min() == pytest.approx(stats.norm.ppf(0.05))
    assert x.max() == pytest.approx(1)


def test_log_scale_density_is_on_the_transformed_scale(layer_data):
    df = pd.DataFrame({"g": ["a"], "d": [stats.lognorm(0.5)]})
    d = layer_data(ggplot(df, aes(y="g", xdist="d")) + stat_slab(normalize="none") + scale_x_log10())
    s = slab(d)
    # on log10(x), lognormal is normal with sd 0.5 / ln(10)
    t = np.sort(s["x"].to_numpy())
    pdf = s.sort_values("x")["pdf"].to_numpy()
    assert np.trapezoid(pdf, t) == pytest.approx(1, abs=0.01)
    assert t[np.argmax(pdf)] == pytest.approx(0, abs=0.01)


def test_discrete_distribution_slab_is_a_step_function(layer_data):
    df = pd.DataFrame({"g": ["a"], "d": [stats.poisson(2)]})
    s = slab(layer_data(ggplot(df, aes(y="g", xdist="d")) + stat_slab(normalize="none")))
    at_2 = s[(s["x"] > 1.5) & (s["x"] < 2.5)]["pdf"]
    assert np.allclose(at_2, stats.poisson(2).pmf(2))


def test_vertical_orientation_is_detected_and_flipped(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("g", "x")) + stat_eye())
    assert (d["orientation"] == "vertical").all()
    iv = interval(d)
    assert {"ymin", "ymax"} <= set(iv.columns)
    assert (iv["ymin"] < iv["y"]).all() and (iv["y"] < iv["ymax"]).all()


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (aes(x="x"), "horizontal"),
        (aes(y="x"), "vertical"),
        (aes(y="g", xdist="d"), "horizontal"),
        (aes(x="g", ydist="d"), "vertical"),
    ],
)
def test_orientation_detection(layer_data, mapping, expected):
    data = DISTS if "xdist" in mapping or "ydist" in mapping else SAMPLES
    d = layer_data(ggplot(data, mapping) + stat_halfeye())
    assert d["orientation"].iloc[0] == expected


def test_orientation_aliases(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye(orientation="y"))
    assert d["orientation"].iloc[0] == "horizontal"
    with pytest.raises(PlotnineError, match="orientation"):
        layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye(orientation="sideways"))


def test_weights_shift_the_median(layer_data):
    df = SAMPLES[SAMPLES["g"] == "a"].assign(w=lambda d: np.where(d["x"] > 0, 5.0, 1.0))
    plain = interval(layer_data(ggplot(df, aes("x", "g")) + stat_pointinterval()))
    weighted = interval(layer_data(ggplot(df, aes("x", "g", weight="w")) + stat_pointinterval()))
    assert weighted["x"].iloc[0] > plain["x"].iloc[0]


def test_pointinterval_has_no_slab(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_pointinterval())
    assert set(d["datatype"]) == {"interval"}


def test_interval_colours_by_level(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_interval())
    assert d["color"].nunique() == 3


def test_ccdf_thickness_and_expansion(layer_data):
    p = ggplot(SAMPLES, aes("x", "g")) + stat_ccdfinterval()
    s = slab(layer_data(p))
    np.testing.assert_allclose(s["thickness"], 1 - s["cdf"])  # normalize="none"
    # expanded to the scale limits: full thickness on the left (1 - 1/n at the
    # leftmost group's minimum value under the empirical CDF)
    assert s.groupby("group")["thickness"].max().tolist() == pytest.approx([1, 1, 1], abs=5e-3)


def test_needs_values(layer_data):
    with pytest.raises(PlotnineError, match="needs sample values"):
        layer_data(ggplot(SAMPLES, aes(color="g")) + stat_halfeye())


# geom layout ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "lo", "hi"),
    [("top", 0.0, 0.9), ("bottom", -0.9, 0.0), ("both", -0.45, 0.45)],
)
def test_side_sets_the_slab_space(layer_data, side, lo, hi):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye(side=side))
    s = slab(d)
    np.testing.assert_allclose(s["ymin"] - s["y"], lo)
    np.testing.assert_allclose(s["ymax"] - s["y"], hi)


def test_justification_and_scale(layer_data):
    s = slab(layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye(justification=0.25, scale=0.5)))
    np.testing.assert_allclose(s["ymin"] - s["y"], -0.125)
    np.testing.assert_allclose(s["ymax"] - s["y"], 0.375)


@pytest.mark.parametrize("normalize", ["all", "panels", "xy", "groups", "none"])
def test_normalize(layer_data, normalize):
    df = pd.concat([SAMPLES.assign(f="p"), SAMPLES.assign(f="q", x=SAMPLES["x"] * 3)])
    d = layer_data(ggplot(df, aes("x", "g")) + stat_slab(normalize=normalize) + facet_wrap("f"))
    t = slab(d).groupby(["PANEL", "group"])["thickness"].max()
    if normalize == "none":
        assert t.max() < 1
    elif normalize == "all":
        assert t.max() == pytest.approx(1)
        assert t.min() < 0.5  # wider panel q has lower densities
    elif normalize == "panels":
        assert t.groupby(level="PANEL").max().tolist() == pytest.approx([1, 1])
    else:  # xy and groups: per slab here
        assert t.tolist() == pytest.approx([1] * len(t))


def test_interval_linewidths_follow_width(layer_data):
    iv = interval(layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye(width=[0.5, 0.8, 0.95])))
    by_width = iv.groupby(".width")["linewidth"].first()
    assert by_width[0.5] > by_width[0.8] > by_width[0.95]
    # points: fatten_point x the thickest interval (x a calibration to ggdist)
    from plotnine_ggdist.geom_slabinterval import _POINT_SCALE

    assert iv["point_size"].iloc[0] == pytest.approx(by_width.max() * 1.8 * _POINT_SCALE)


def test_default_sizes_match_ggdist():
    from plotnine._utils import SIZE_FACTOR

    from plotnine_ggdist.geom_slabinterval import DEFAULT_INTERVAL_SIZE_RANGE, _POINT_SCALE

    # ggplot2 linewidth w is drawn w * 72.27 / 25.4 * 72 / 96 points wide
    thin, thick = (v * SIZE_FACTOR for v in DEFAULT_INTERVAL_SIZE_RANGE)
    assert thin == pytest.approx(0.6 * 2.1343, rel=1e-3)
    assert thick == pytest.approx(1.4 * 2.1343, rel=1e-3)
    # default point: about 7 pt across, outline included
    size = DEFAULT_INTERVAL_SIZE_RANGE[1] * 1.8 * _POINT_SCALE
    assert (size + 0.5) * SIZE_FACTOR + 0.5 * SIZE_FACTOR == pytest.approx(7.06, abs=0.1)


def test_set_linewidth_and_point_size_are_respected(layer_data):
    iv = interval(layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_pointinterval(linewidth=3, point_size=7)))
    assert (iv["linewidth"] == 3).all()
    assert (iv["point_size"] == 7).all()


def test_geom_pointinterval_uses_unmapped_width_column(layer_data):
    summary = median_qi(SAMPLES, "x", by="g", width=[0.66, 0.95])
    d = layer_data(ggplot(summary, aes(x="x", xmin=".lower", xmax=".upper", y="g")) + geom_pointinterval())
    assert d["orientation"].iloc[0] == "horizontal"
    assert d.groupby(".width")["linewidth"].first().is_monotonic_decreasing


def test_geom_interval_colours_by_width(layer_data):
    summary = median_qi(SAMPLES, "x", by="g", width=[0.5, 0.95])
    d = layer_data(ggplot(summary, aes(y="x", ymin=".lower", ymax=".upper", x="g")) + geom_interval())
    assert d["orientation"].iloc[0] == "vertical"
    assert d["color"].nunique() == 2


def test_geom_slab_orientation_from_discrete_axis(layer_data):
    df = pd.DataFrame({"g": np.repeat(["a", "b"], 50), "v": np.tile(np.linspace(-3, 3, 50), 2)})
    df["t"] = stats.norm.pdf(df["v"])
    h = layer_data(ggplot(df, aes(y="g", x="v", thickness="t")) + geom_slab())
    v = layer_data(ggplot(df, aes(x="g", y="v", thickness="t")) + geom_slab())
    assert h["orientation"].iloc[0] == "horizontal"
    assert v["orientation"].iloc[0] == "vertical"


def test_coord_flip_builds(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye() + coord_flip())
    assert len(d)


# lineribbon --------------------------------------------------------------------------


LINE = pd.DataFrame({"x": np.repeat(np.arange(5), 200)})
LINE["y"] = rng.normal(LINE["x"], 1)


def test_stat_lineribbon(layer_data):
    d = layer_data(ggplot(LINE, aes("x", "y")) + stat_lineribbon())
    assert len(d) == 5 * 3
    expected = median_qi(LINE, "y", by="x", width=[0.5, 0.8, 0.95])
    np.testing.assert_allclose(
        d.sort_values(["x", ".width"])["ymax"].to_numpy(),
        expected.sort_values(["x", ".width"])[".upper"].to_numpy(),
    )
    assert d["fill"].nunique() == 3


def test_stat_lineribbon_distributions(layer_data):
    df = pd.DataFrame({"x": [1, 2, 3], "d": [stats.norm(i, 1) for i in (1, 2, 3)]})
    d = layer_data(ggplot(df, aes(x="x", ydist="d")) + stat_lineribbon(width=[0.95]))
    np.testing.assert_allclose(d["ymax"] - d["y"], 1.959964, atol=1e-5)


def test_geom_lineribbon_fills_by_width_column(layer_data):
    summary = median_qi(LINE, "y", by="x", width=[0.5, 0.95])
    d = layer_data(ggplot(summary, aes(x="x", y="y", ymin=".lower", ymax=".upper")) + geom_lineribbon())
    assert d["fill"].nunique() == 2


def test_geom_lineribbon_user_fill_wins(layer_data):
    summary = median_qi(LINE, "y", by="x", width=[0.5, 0.95])
    d = layer_data(
        ggplot(summary, aes(x="x", y="y", ymin=".lower", ymax=".upper"))
        + geom_lineribbon(fill="red")
    )
    assert (d["fill"] == "red").all()


def test_after_stat_expressions(layer_data):
    d = layer_data(ggplot(SAMPLES, aes("x", "g")) + stat_halfeye(aes(fill=after_stat("x > 0"))))
    assert slab(d)["fill"].nunique() == 2
