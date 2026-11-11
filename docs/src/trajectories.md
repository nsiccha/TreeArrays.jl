# Concentration trajectories

Summarize simulated concentration curves on subject-specific observation grids and plot their pointwise intervals. The data are synthetic parameter draws from a lognormal distribution, with concentration in mg/L and time in hours. They illustrate postprocessing; they are neither a fitted posterior nor measurements from patients. Each subject has a different time grid. The draw index identifies a complete trajectory, so reducing time within a draw preserves the relationship between its observations.

## Data

```@example trajectories
using TreeArrays, Random, Tables, Statistics
rng = MersenneTwister(42)
draws = 1:128
subjects = ["A", "B"]
grids = [collect(0.0:0.5:8.0), collect(0.0:1.0:8.0)]
rates = [0.25, 0.45] # elimination rate in h⁻¹
leaves = map(grids, rates) do time, rate
    initial = 10 .* exp.(0.12 .* randn(rng, length(draws)))
    elimination = rate .* exp.(0.15 .* randn(rng, length(draws)))
    concentration = [c * exp(-k * t) for (c, k) in zip(initial, elimination), t in time]
    TreeData(concentration, :draw => draws, :time => time)
end
curves = TreeData(leaves, :subject => subjects)
[(; subject=s, draws=size(x, :draw), observations=size(x, :time))
 for (s, x) in zip(subjects, leaves)]
```

Each leaf stores a draw-by-time matrix. The outer tree attaches subject labels without padding the shorter grid or interpolating observations. The generated sizes above describe the data that actually entered the calculation. In a fitted analysis, replace these leaves with draws from the model and retain the model's draw and subject identifiers.

## Pointwise intervals

Reduce `:draw` while keeping each subject's time grid. A named record of probabilities makes columns for the endpoints and median. The table below displays the first subject's early observations; the full table feeds the figure.

```@example trajectories
probabilities = (lower=0.05, median=0.5, upper=0.95)
bands = quantile(curves, probabilities; dims=:draw)
rows = Tables.rowtable(bands)
first(rows, 4)
```

These are pointwise intervals: at each time, they describe the distribution across simulated curves. They do not give a simultaneous band covering a specified fraction of whole trajectories. Taking the median at each time also does not select one particular draw. The endpoints remain attached to the observation coordinates that produced them, including the difference between subjects' grids.

Use the long table interface for iteration over observations. Record fields become columns, so a plotting function can read `lower`, `median`, and `upper` without reconstructing the tree. The [interoperability guide](tables.md) covers the alternative of making an explicit band axis and choosing a wide view later.

## Figure

The line shows the pointwise median; the shaded band spans the probabilities defined above. Both subjects use the same concentration and time scales. Subject B's larger simulated elimination rate produces a steeper decline; the plot makes that difference visible without interpreting it as a treatment effect.

```@example trajectories
using AlgebraOfVega: data, mapping, lineribbon, config, vdraw, scales, vega_head
plot = data(rows) *
    mapping(:time => "Time (h)", :median => "Concentration (mg/L)";
            color=:subject => "Subject") *
    lineribbon(bands=[:lower => :upper]) *
    config(width=520, height=300, theme=:host,
           config=(; axis=(; labelFontSize=13, titleFontSize=14)))
figure = vdraw(plot, scales(Color=(; palette=["#2a78d6", "#eb6834"])))
head = vega_head(source=:vendor, base="./assets/aov", runtime=:linked)
figure_html = join(sprint(show, MIME"text/html"(), node) for node in [head; figure]) # hide
write(joinpath(pkgdir(TreeArrays), "docs", "build", "trajectory-figure.html"), figure_html) # hide
HTML("<div data-figure-src='assets/trajectory-figure.html'></div><script></script>") # hide
```

The ribbon joins neighboring observation times for display. It does not add observations to either input grid. Compare subjects at shared times when the comparison needs observed coordinates; an interpolation rule would be an additional modeling choice. The colors identify subjects throughout the figure, and the axes state their physical units.

## Exposure

Integrate each draw over its own time grid before summarizing across draws. This order retains within-trajectory dependence. Integrating the median curve would answer a different question because integration and quantiles do not generally commute. The trapezoidal rule uses the supplied coordinates rather than assuming equally spaced observations.

```@example trajectories
trapz(t, y) = sum((t[i+1] - t[i]) * (y[i+1] + y[i]) / 2
                  for i in 1:length(t)-1)
exposure = mapslices(curves; dims=:time, coords=true) do y, t
    trapz(t, y)
end
exposure_bands = quantile(exposure, probabilities; dims=:draw)
Tables.rowtable(exposure_bands)
```

Exposure has units of mg·h/L over the displayed time window. This summary keeps subject labels but removes time from the varying axes. The underlying per-draw exposures remain available if a later calculation needs paired transformations or a different interval definition. The [reductions guide](reductions.md) explains coordinate-aware kernels and pooled dimensions.

## Checks

Check structural and numerical properties before consuming the summaries. Every pointwise median must lie between its endpoints, all exposure values must be finite, and the rows for each subject must retain that subject's original grid. These checks are about this postprocessing calculation. They do not diagnose sampling convergence or validate the model that supplied the draws.

```@example trajectories
checks = (
    ordered_intervals=all(r.lower <= r.median <= r.upper for r in rows),
    finite_exposure=all(isfinite(r.value) for r in Tables.rows(exposure)),
    grids_preserved=all([r.time for r in rows if r.subject == s] == t
                        for (s, t) in zip(subjects, grids)),
)
@assert all(values(checks))
checks
```

The synthetic generator has no Markov chains, so sampler diagnostics do not apply. For sampled inputs, assess convergence before combining chains, and use a [numeric array view](tables.md#Numeric-arrays) when a diagnostic requires an `AbstractArray`. For unequal observation grids, [coordinate alignment](ragged.md) is an explicit choice whenever an outer reduction keeps time.

## Reproduce

::: details Reproduce

Run these shell commands from a checkout with Julia 1.12 or newer. Dependency setup precedes the build;
these shell instructions are not executed as documentation examples.

```sh
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
```

The build generates the figure and this receipt. No fit or downloaded dataset is required.

```@example trajectories
using SHA
import AlgebraOfVega
source = joinpath(pkgdir(TreeArrays), "docs", "src", "trajectories.md")
(; julia=string(VERSION), treearrays=string(pkgversion(TreeArrays)),
   algebraofvega=string(pkgversion(AlgebraOfVega)),
   source_sha256=bytes2hex(sha256(read(source))), seed=42)
```

:::
