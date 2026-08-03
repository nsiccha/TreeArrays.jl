# TreeArrays.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/TreeArrays.jl/dev/)

**Named, hierarchical, possibly-ragged arrays over flat backing storage**, plus
streaming reductions that compute summary statistics over named axes.

TreeArrays is a **postprocessing layer** for Bayesian posterior draws and PKPD
predictions. You already have the draws — from a sampler, an ODE solve, a
prediction sweep. TreeArrays names their axes, lets you reduce over those axes by
name, and hands the result to a plotting or diagnostics layer without ever
materializing the cross-product of everything.

📖 **[Documentation](https://nsiccha.github.io/TreeArrays.jl/dev/)**

## Installation

Not registered — add it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/TreeArrays.jl")
```

Hard dependencies: `Statistics` and `Tables`. `NaNStatistics` is a weak
dependency enabling NaN-safe quantiles.

## The idea in one example

```julia
using TreeArrays, Statistics

X = TreeData(draws, :draw, :subject, :time => times)   # name the axes

quantile(                                     # 3. reduce :draw    → a ribbon axis
  quantile(                                   # 2. reduce :subject → a percentile axis
    mapslices(maximum, X; dims = :time),      # 1. reduce :time    → one number per (draw, subject)
    TreeDim(:percentile, (0.05, 0.5, 0.95)); dims = :subject),
  TreeDim(:ribbon, (0.025, 0.5, 0.975)); dims = :draw)
```

Each step collapses only the axes it names, keeps every other axis and its
labels, and stays a `TreeData`. Nothing is stacked, pivoted or `vcat`ed in
between — and the fat pre-reduction frame that a `groupby`/`combine`/`vcat`
pipeline would build simply never exists.

## The one load-bearing invariant

> **Eager compute / lazy assembly — no eager restructuring, ever.**

Arithmetic runs *now*, in tight type-stable loops, on each materialized block.
What is lazy is *forming the big combined matrix*: the cross-product of
categorical and virtual axes is described structurally, and a reduction
materializes one block, runs the kernel, discards it, and moves on.

This is deliberately **not** a deferred-compute / `BroadcastArray` design. The
performance enemies here are allocating the big combined matrix and poor
per-block kernel quality — not the cost of the individual evaluations.

## What you get

- **Metadata lives once, as an axis.** A fixed dim (`dose = 20`) is stored once,
  O(1) — never smeared across N rows.
- **Ragged is first class.** Per-subject series of differing length are a tree of
  sub-arrays over shared backing storage, and the *same* reduction code runs on
  rectangular and ragged input.
- **One-pass quantiles.** `quantile(X, TreeDim(:band, ps); dims = :draw)` computes
  every level in a single sorted pass and gives you an output axis — never loop
  per level.
- **Kernels can see their axis.** `coords = true` (or a 2-argument `@kernel`)
  hands the kernel the reduced axis's coordinates, bound *per node* — so on ragged
  data every subject's call sees its own grid. AUC and `tmax` just work.
- **Zero-copy selection.** `selectdim(X, :param => r"^beta")` slices the backing
  array with a `view`; a subset of a `(draw × chain × 600_000)` matrix never
  materializes.
- **A Tables.jl source, for free.** Records melt wide, axes melt long, and the
  columns are lazy views — `@allocated` stays flat across 100× rows. Plot it
  without ever building a DataFrame.
- **Rich display, no dependency.** `TreeData` and `TreeTable` render as HTML *and*
  markdown out of the box (`show`/`MIME` are Base), so they drop into a web route,
  Pluto or Jupyter directly. Every preview is bounded and every elision marked.
- **Never silently wrong.** `dims =` is *foundALL*: a name that resolves nowhere
  throws. Unsupported shapes error by name rather than returning something
  plausible-looking.

## How this relates to DimensionalData.jl

Heavy overlap — named dims, dim-aware reductions, a Tables.jl bridge — and
TreeArrays deliberately borrows DD's central good idea: dim *identity* lives in
the type, so axis resolution is compile-time and reductions stay type-stable.

Two differentiators drove building something separate:

1. **Lazy assembly, not lazy compute.** DD's operations are eager *and*
   materializing. TreeArrays keeps arithmetic eager but never allocates the
   combined matrix.
2. **Ragged axes.** DimensionalData is fundamentally rectangular; subject-varying
   lengths are the normal case here.

The honest cost: a `DimArray` **is** an `AbstractArray` and drops into the whole
array ecosystem for free, whereas a `TreeData` deliberately is not one — a record
node and a ragged node are not arrays in any honest sense. Crossing into
array-land is an explicit, zero-copy step (`TreeActualArray(X)`).

**Will TreeArrays supersede DD?** Open, and stated as such rather than promised.
The intent is that TreeArrays eventually covers everything *its own* users need
from a named-dimension array. It is not intended to subsume DD's much broader
scope (rasters, climate cubes, geospatial stacks) — and a package that is not an
array cannot strictly supersede one that is. See
[Comparisons](https://nsiccha.github.io/TreeArrays.jl/dev/comparisons) for the
full discussion, including a sketch of how TreeArrays could interoperate with
**FlexiChains.jl**.

## Status

`v0.1.0`, unregistered, young. Used in anger for PKPD postprocessing, but the API
is not frozen and some corners are explicitly unfinished (`setdim` is a stub).
Where a limit exists, the error says so.
