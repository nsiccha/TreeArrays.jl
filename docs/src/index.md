```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
# NOTE: this frontmatter MUST stay inside a `@raw html` fence. Documenter parses
# a bare `---` block as ordinary markdown (a thematic break plus paragraphs),
# which collapses the YAML indentation and ships the keys to the page as visible
# body text -- VitePress never sees frontmatter at all.
layout: home

hero:
  name: TreeArrays.jl
  text: Named, hierarchical, possibly-ragged arrays
  tagline: A postprocessing layer for Bayesian posterior draws and PKPD predictions — eager compute, lazy assembly, no eager restructuring, ever.
  actions:
    - theme: brand
      text: Getting started
      link: /getting-started
    - theme: alt
      text: Reductions
      link: /reductions
    - theme: alt
      text: vs. DimensionalData
      link: /comparisons
    - theme: alt
      text: View on GitHub
      link: https://github.com/nsiccha/TreeArrays.jl

features:
  - title: Metadata lives once, as an axis
    details: A dim label is stored once and reused, never smeared across N rows. The fat pre-reduction frame that a groupby/combine/vcat pipeline builds simply never exists.
  - title: Ragged is first class
    details: Per-subject series of differing length are a tree of sub-arrays over shared backing storage. The same reduction code runs on rectangular and ragged input.
  - title: Lazy assembly, eager compute
    details: The big combined matrix is described structurally and never allocated; reductions stream over it block by block. No deferred compute graph, no per-element closures.
  - title: A Tables.jl source, for free
    details: Records melt wide, axes melt long, and the columns are lazy views — @allocated stays flat across 100× rows. Plot it without ever building a DataFrame.
---
```

## What it is

`TreeArrays.jl` gives you **named, hierarchically-structured, possibly-ragged
arrays over flat backing storage**, plus **streaming reductions** that compute
summary statistics over named axes.

It is built for one job: **postprocessing**. You already have the draws — from a
sampler, from an ODE solve, from a prediction sweep. TreeArrays names their axes,
lets you reduce over those axes by name, and hands the result to a plotting or
diagnostics layer without ever materializing the cross-product of everything.

```julia
using TreeArrays

X = TreeData(draws, :draw, :subject, :time => times)   # name the axes

quantile(                                     # 3. reduce :draw   → a ribbon axis
  quantile(                                   # 2. reduce :subject → a percentile axis
    mapslices(maximum, X; dims = :time),      # 1. reduce :time    → one number per (draw, subject)
    TreeDim(:percentile, (0.05, 0.5, 0.95)); dims = :subject),
  TreeDim(:ribbon, (0.025, 0.5, 0.975)); dims = :draw)
```

Each step keeps every axis it did not reduce, keeps the labels, and stays a
`TreeData`. Nothing is stacked, pivoted or `vcat`ed in between.

## The one load-bearing invariant

> **Eager compute / lazy assembly — no eager restructuring, ever.**

Arithmetic runs *now*, in tight type-stable loops, on each materialized block.
What is lazy is *forming the big combined matrix*: the cross-product of
categorical and virtual axes is described structurally, and a reduction
materializes one block, runs the kernel, discards it, and moves on.

This is deliberately **not** a deferred-compute / `BroadcastArray` design. The
performance enemies here are allocating the big combined matrix and poor
per-block kernel quality — not the cost of the individual evaluations.

## Status

Version `0.1.0`, unregistered, and young. It is used in anger for PKPD
postprocessing, but the API is not frozen and some corners are explicitly
unfinished — [`setdim`](@ref) is still a stub, and a handful of shapes error by
name rather than being supported. Where a limit exists, the error says so; the
package tries hard never to return a plausible-looking wrong answer.

Install it straight from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/TreeArrays.jl")
```
