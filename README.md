# plotnine-ggdist

Distribution and uncertainty visualization for [plotnine](https://plotnine.org):
half-eye and eye plots, dot plots and quantile dot plots, beeswarms,
point + multiple-interval plots, CCDF bars, gradient plots, histograms
with intervals, and line + ribbon plots, from
either **sample data** (posterior draws, bootstrap samples) or **analytical
distributions** (`scipy.stats` frozen distributions).

A Python counterpart of [ggdist](https://mjskay.github.io/ggdist/) by
Matthew Kay. It is an independent implementation, written from ggdist's
documentation, its papers and its rendered output (see [License](#license)).
It is a sibling of [plotnine-ggtext](https://github.com/snake-case-ggtext/plotnine-ggtext).

![Half-eye plot of posterior draws and a lineribbon forecast from scipy distributions](demo/demo.png)

<sub>Made by [`demo/demo.py`](demo/demo.py).</sub>

## Install

```bash
pip install git+https://github.com/snake-case-ggtext/plotnine-ggdist
```

Requires Python ≥ 3.11, plotnine ≥ 0.15 and scipy.

## Usage

```python
import numpy as np
import pandas as pd
from scipy import stats
from plotnine import aes, after_stat, ggplot
from plotnine_ggdist import stat_halfeye, stat_pointinterval, median_qi

rng = np.random.default_rng(1)
draws = pd.DataFrame({
    "group": np.tile(["a", "b", "c"], 500),
    "value": rng.normal(np.tile([5, 7, 9], 500), np.tile([1, 1.5, 1], 500)),
})

# from samples
ggplot(draws, aes(x="value", y="group")) + stat_halfeye()

# from analytical distributions: a column of frozen scipy.stats distributions
dists = pd.DataFrame({"group": ["a", "b", "c"],
                      "dist": [stats.norm(5, 1), stats.t(3, 7), stats.gamma(8)]})
ggplot(dists, aes(y="group", xdist="dist")) + stat_halfeye(aes(fill=after_stat("level")))

# quantile dot plots
ggplot(dists, aes(y="group", xdist="dist")) + stat_dotsinterval(quantiles=50)

# summaries as data frames
median_qi(draws, "value", by="group", width=[0.66, 0.95])
draws.groupby("group").pipe(median_qi, "value")
```

### A tour of every feature

`demo/tour.py` draws an example of each feature (about 40 plots) into
`demo/tour_output/`; `demo/tour_marimo.py` is the same tour as a
[marimo](https://marimo.io) notebook, one plot per cell:

```bash
uv run python demo/tour.py
uv run --with marimo marimo edit demo/tour_marimo.py
```

## What's included

| ggdist (R) | plotnine-ggdist | |
|---|---|---|
| `stat_slabinterval()`, `geom_slabinterval()` | same | the general slab + interval stat / geom |
| `stat_halfeye()`, `stat_eye()` | same | density + point + intervals |
| `stat_slab()`, `geom_slab()` | same | density only; ridgelines with `height`, `expand`, `trim` |
| `stat_pointinterval()`, `geom_pointinterval()` | same | point + multiple intervals |
| `stat_interval()`, `geom_interval()` | same | intervals coloured by width |
| `stat_ccdfinterval()`, `stat_cdfinterval()` | same | (C)CDF bar plots |
| `stat_gradientinterval()` | same | opacity follows density |
| `stat_histinterval()` | same | histogram + intervals |
| `stat_lineribbon()`, `stat_ribbon()`, `geom_lineribbon()` | same | line + ribbons |
| `stat_dotsinterval()`, `geom_dotsinterval()` | same | dot plot + point + intervals |
| `stat_dots()`, `geom_dots()` | same | dot plots; quantile dot plots with `quantiles=` (always, for distributions) |
| `geom_swarm()`, `geom_weave()` | same | beeswarm-style dot plots |
| `geom_blur_dots()`, `stat_mcse_dots()` | same | blurry dots, e.g. Monte Carlo error of each quantile |
| `blur_gaussian()`, `blur_interval()` | same | blur functions for `blur=` |
| `smooth_none()`, `smooth_bounded()`, `smooth_unbounded()`, `smooth_discrete()`, `smooth_bar()` | same | dot smoothers for `smooth=`; call with options only to set them, e.g. `smooth_discrete(kernel="ep")` |
| `posterior::mcse_quantile()`, `posterior::ess_basic()` | `mcse_quantile()`, `ess_basic()` | one chain, draws in data order |
| `bin_dots()`, `find_dotplot_binwidth()` | same | the dot layout, for your own plots |
| `grid::unit()` | `unit()` | physical sizes for `binwidth`, e.g. `unit(1.5, "mm")` |
| `point_interval()`, `median_qi()`, `mean_hdi()`, … (15 shortcuts) | same | samples, distributions, DataFrames, `groupby` |
| `qi()`, `hdi()`, `hdci()`, `ll()`, `ul()`, `Mode()` | same | |
| `density_bounded()`, `density_unbounded()`, `density_histogram()` | same | |
| `bandwidth_dpi()`, `bandwidth_nrd0()`, `bandwidth_nrd()` | same | `dpi` (Sheather–Jones plug-in) is the default |
| `bounder_cdf()`, `bounder_cooke()`, `bounder_range()` | same | |
| `weighted_quantile()`, `weighted_ecdf()` | same | |
| `theme_ggdist()` | same | |
| `distributional::dist_normal(…)` | `scipy.stats.norm(…)` | any frozen `scipy.stats` distribution, continuous or discrete |

The computational core (`point_interval()`, the interval functions and the
density estimators) does not depend on plotnine, so it can be used with
plain matplotlib, seaborn or anything else.

### Aesthetics

The slab, interval and point sub-geometries each have their own
aesthetics that fall back to the shared ones, as in ggdist:

| sub-geometry | aesthetics | falls back to |
|---|---|---|
| slab | `slab_fill`, `slab_color`, `slab_alpha`, `slab_linewidth`, `slab_linetype` | `fill`, (none), `alpha`, –, `linetype` |
| interval | `interval_color`, `interval_alpha`, `interval_linetype`, `linewidth` | `color`, `alpha`, `linetype` |
| point | `point_color`, `point_fill`, `point_alpha`, `point_size` | `color`, `point_color`, `alpha`, `size` |
| dots | `slab_shape` (matplotlib marker or R `pch` 0-25), `order` (stacking order), `sd` (blur dots) | `shape` for `geom_dots()`, else `"o"` |
| layout | `thickness`, `side`, `scale`, `justification`, `height` | |

Computed variables for `after_stat()`: `pdf`, `cdf`, `n`, `level`, `.width`,
and `x`/`y`, `xmin`/`xmax` or `ymin`/`ymax`. For dots, `level` and `.width`
are those of the narrowest interval containing each dot, so
`aes(fill=after_stat("level"))` colours dots by interval.

### Dot layouts

The dots are laid out when the plot is drawn: the bin width is chosen so
the tallest stack fits the space each group gets (`side`, `scale`,
`height`), with one dot size per panel. `layout=` picks the arrangement:

- `"bin"`: Wilkinson dot-density bins, formed from the centre of the data
  outwards so symmetric data give a symmetric plot; overlapping bins are
  nudged apart (`overlaps="nudge"`) by the smallest weighted least-squares
  shift.
- `"weave"`: rows as in `"bin"`, but each dot stays at its own value.
- `"hex"`: alternate rows offset by a quarter bin width; pair with
  `stackratio=0.9` for hexagonal packing.
- `"swarm"`: a compact beeswarm (for more than 400 dots, a faster
  one-pass version).
- `"bar"`: for discrete data; each value's dots fill a rectangular bar,
  row by row (use `order=` to stack, e.g., by a fill variable, and
  `group=1` to lay out every group together).

`smooth=` moves dot positions before the layout: `"unbounded"`/`"bounded"`
replace them with evenly spaced quantiles of a density estimate (a
"density dotplot"), `"discrete"` and `"bar"` spread duplicated values.

A fixed `binwidth` (data units, or a `unit()`) that makes stacks too tall
warns by default; `overflow="compress"` squeezes the stacks instead.

## Differences from ggdist

plotnine and Python conventions are used where they differ from ggplot2
and R:

| ggdist | plotnine-ggdist | note |
|---|---|---|
| `.width = c(.66, .95)` | `width=(0.66, 0.95)` | a leading dot isn't valid in a Python keyword |
| `after_stat(.width)` | `after_stat("level")` | `".width"` can't be used in a plotnine expression; `level` is the same as an ordered categorical. Plain mappings like `aes(xmin=".lower")` work |
| `group_by(g) \|> median_qi(x)` | `median_qi(df, "x", by="g")` or `df.groupby("g").pipe(median_qi, "x")` | |
| `median_qi(.width = .9)` passed to a stat | `point_interval="median_qi"`, `width=0.9` | or a `(point, interval)` tuple of names or functions |
| `xdist = dist_normal(mu, sigma)` | `xdist="col"` holding `stats.norm(mu, sigma)` | build the column first: `df["d"] = [stats.norm(m, s) for m, s in …]` |
| `posterior::rvar()` | not yet | pass draws as long-format samples |
| `slab_type = "pdf"` (deprecated in ggdist) | not supported | map `thickness=after_stat("cdf")` etc. instead |
| line widths in mm | plotnine units (`value × √π` points) | defaults are converted so intervals look the same as ggdist's (0.6–1.4 mm) |
| `orientation = NA` | `orientation=None` | `"horizontal"`/`"y"`, `"vertical"`/`"x"` as in ggdist |
| `fill_type = "gradient"` needs R ≥ 4.1 and device support | works everywhere | an image clipped to the slab |
| `breaks = "Scott"` (with `pretty()`) | `breaks="scott"` | numpy's bin-edge rules; bins can differ slightly |
| `stat_interval()` default colours | viridis (plotnine's ordinal default) | add `scale_color_brewer()` for ggdist's examples' look |
| dots `binwidth = unit(1.5, "mm")` | `binwidth=unit(1.5, "mm")` | plotnine-ggdist's own `unit()`; bare numbers are data units, as in ggdist |
| `slab_shape = 21` (pch codes) | `slab_shape=21`, or matplotlib markers (`"o"`, `"s"`, `"^"`, ...) | pch 7 and 9-14 (composite symbols) are drawn as their base shape; font glyphs (`family`) aren't supported |
| `stat_mcse_dots()` on multi-chain `rvar`s | draws as one chain | |
| `density = "bounded"` with `adapt` | no `adapt` | all of `stats::density()`'s kernels are available via `kernel=` |

Not in this release (planned): `geom_spike()`, `position_dodgejust()`
(plotnine's `position_dodge()` works for vertical layouts only), colour
ramp scales, `subguide_*()`, `parse_dist()`, `curve_interval()`,
`cut_cdf_qi()`, and `posterior::rvar()`/ArviZ input.

## Tests

```bash
uv run pytest
uvx ruff check .   # lint, as CI does
```

- `tests/test_intervals.py`, `tests/test_density.py`: the computational core
  against known values (scipy quantiles, analytic HDIs and modes, KDE mass
  and boundary behaviour).
- `tests/test_stat_geom_data.py`: what the stats compute and how the geoms
  lay it out, checked on the built layer data.
- `tests/test_dots.py`: the dot layouts (no overlaps, symmetry, fitting the
  space), quantile dot plots, and the bin width chosen at draw time.
- `tests/test_smooth_blur_shapes.py`: kernels, smoothers, the bar layout,
  pch shapes, blurs, and MCSE/ESS against their theoretical values.
- `tests/test_examples.py`: image tests. Every example in ggdist's reference
  pages for the included functions has a one-to-one port (named after the
  reference image, e.g. `stat_halfeye-1`), plus tests for features the
  examples don't cover. Because numpy can't reproduce R's random numbers,
  the draws live in `tests/data/` (`python tests/data/make_data.py`).

`tests/reference_r/render.R` renders the same plots, from the same data, with
R's ggdist for side-by-side comparison:

```bash
Rscript tests/reference_r/render.R   # needs ggdist, distributional, dplyr, posterior, ragg
```

To update baselines after an intended visual change:

```bash
uv run pytest --update-baselines
```

Baselines must be generated on Linux (CI runs on Ubuntu): font rendering
differs on macOS and Windows, so images made there fail in CI. On a Mac,
regenerate them in Docker, e.g.

```bash
docker run --rm -v "$PWD":/repo -w /repo ghcr.io/astral-sh/uv:python3.14-trixie \
  uv run pytest --update-baselines
```

or let CI do it: when image tests fail, the workflow uploads the images it
made as an artifact.

## License

MIT.

ggdist is GPL-3. To keep this package permissively licensed, it does not
translate ggdist's R code: it was written from ggdist's documentation and
reference pages, the papers it cites (e.g. Wand & Jones 1995 for the
plug-in bandwidth, Cooke 1979 for bounds estimation, Wilkinson 1999 for
dot plots, Vehtari et al. 2021 for MCSE, Kay et al. 2016 and Fernandes et
al. 2018 for the plot types), and comparisons of rendered
output. The R reference images are not included in the repository; render
them with `tests/reference_r/render.R`.
