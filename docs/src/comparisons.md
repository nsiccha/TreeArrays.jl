```@meta
CurrentModule = TreeArrays
```

# Comparisons

TreeArrays sits next to two packages people reasonably ask about:
**DimensionalData.jl**, which also gives arrays named dimensions, and
**FlexiChains.jl**, which holds MCMC output. This page says what is genuinely
different, what is borrowed, and — for FlexiChains — what the shipped
integration does.

## [TreeArrays vs. DimensionalData](@id vs-dd)

The overlap is heavy and deliberate: named dims, dim-aware reductions,
selectors, a Tables.jl bridge. TreeArrays also **borrows DD's central good
idea** — dim *identity* lives in the type, so axis resolution happens at compile
time and reductions stay type-stable.

Two differentiators drove building something separate.

### (a) Lazy *assembly* — not lazy compute

DimensionalData's operations are eager **and** they materialize: a reduction
produces a new `DimArray` holding real memory, and the array you reduce is a real
array to begin with.

TreeArrays keeps the arithmetic eager and makes the **assembly** lazy. The
cross-product of categorical and virtual axes — draws × subjects × doses ×
placebo × parameter-space — is described *structurally* and never allocated. A
reduction materializes one block, runs the kernel on it, discards it, and moves
to the next.

This is a narrow claim, and it is worth being precise about what it is *not*:

> TreeArrays is **not** a deferred-compute package. There is no `BroadcastArray`,
> no compute graph, no per-element closure, no recompute-on-reaccess. Transforms
> and summary statistics run *now*, in tight type-stable loops. The only thing
> that is lazy is *forming the big combined matrix*.

The performance enemies in this domain are allocating that combined matrix and
poor per-block kernel quality — not the cost of individual evaluations. A
deferred-compute design attacks the wrong one and usually loses.

### (b) Ragged axes

DimensionalData is fundamentally **rectangular**. Every element of a dimension has
the same shape underneath it.

The data this package was written for is not: 179 subjects with different numbers
of measurements, and a *separate* irregular dose history per subject. In
TreeArrays that is a [`TreeRaggedArray`](@ref) — an outer axis of sub-trees with
different shapes — and the *same* `mapslices(f, X; dims = :time)` call runs on it,
because rectangular is just ragged with uniform offsets. See
[Ragged data](@ref).

### The honest cost: a `TreeData` is not an `AbstractArray`

This is the trade the table below cannot express. A `DimArray` **is** an
`AbstractArray`, so it drops into the entire Julia array ecosystem for free —
every function typed on `AbstractArray`, every broadcast, every linear-algebra
call.

A `TreeData` is deliberately **not** an `AbstractArray` subtype, because a record
node and a ragged node are not arrays in any honest sense, and pretending
otherwise would make the dispatch lie. The price is that crossing into array-land
is an explicit step: [`TreeActualArray(X)`](@ref) — a lazy, zero-copy view that
also keeps the labels — or `parent(X)` / `collect(X)`. See
[Feeding a numeric-array API](@ref actual-array).

If your data is rectangular and you want it to behave like an array everywhere,
that is a genuine reason to prefer DimensionalData.

### Side by side

| | DimensionalData.jl | TreeArrays.jl |
|---|---|---|
| Core type | `DimArray <: AbstractArray` | `TreeData`, deliberately not an array |
| Shape | rectangular | rectangular **or ragged** |
| Nesting | `DimStack` of co-dimensional layers | a recursive **tree**: leaves, records, ragged nodes |
| Ops | eager, materializing | eager compute, **lazy assembly** |
| Dim identity in the type | yes | yes (borrowed) |
| Selectors | rich (`At`, `Near`, `Between`, …) | `selectdim` — Regex / predicate / mask / indices |
| Scope | general-purpose named arrays; the base of a GIS/raster ecosystem | a narrow **postprocessing / summarization** layer |
| Maturity | mature, widely used | `v0.1.0`, unregistered, young |

### Will TreeArrays supersede DimensionalData?

**Genuinely open, and stated as such rather than promised.** The original design
note lists "build on DD vs. standalone" as the key unresolved question, and the
standalone route was taken to get the two differentiators above — not because DD
was found wanting on its own ground.

What is intended: that TreeArrays eventually covers everything its own users
currently need from a named-dimension array, so a PKPD postprocessing pipeline
never has to reach for both. What is **not** claimed: that it will subsume DD's
scope. DD anchors a much broader ecosystem (rasters, climate data cubes,
geospatial stacks) whose requirements are not on this package's roadmap, and it
gives up nothing to be an `AbstractArray`, which TreeArrays gives up
deliberately. A package that is not an array cannot strictly supersede one that
is.

Treat the two as overlapping-but-distinct until this page says otherwise.

## [TreeArrays and FlexiChains](@id vs-fc)

!!! note "Shipped: this section describes a tested package extension"
    `TreeData(chain)`, `FlexiChain(X)`, and `TreeData(summary)` below are real,
    tested API — extension `TreeArraysFlexiChainsExt`, loaded automatically
    whenever both packages are loaded. Nothing here is a sketch.

FlexiChains.jl and TreeArrays solve **adjacent, non-overlapping** problems.

FlexiChains is a **container for sampler output**: it holds what a sampler
produced, over iterations and chains, and its distinguishing feature is that a
"parameter" need not be a `Float64` — a draw can be a vector, a matrix, or a
custom struct, keyed by the variable names the model actually used. That is a
storage and identity problem.

TreeArrays is a **summarization layer**: given draws that already exist, name
their axes and reduce over them, without ever building the fat intermediate
frame. That is a reduction problem.

Neither replaces the other, and the seam between them is clean.

### The bridge

The two type systems line up almost directly, because a chain object and a
`TreeData` describe the same logical thing — a `(draw, chain)` grid of keyed
values:

| FlexiChains | TreeArrays |
|---|---|
| the iteration dimension | a `:draw` axis, with the chain's index lookups as coordinates |
| the chain dimension | a `:chain` axis, same |
| one scalar parameter key | a field of a [`TreeNamedTuple`](@ref) record over `:draw` + `:chain` |
| one uniformly array-valued parameter key | a nested `TreeData` leaf with its own inner axes (`:elem`, or `:elem_1`, `:elem_2`, …) |
| keys whose cells differ in shape or type | refused by name — split the key upstream |

So ingesting a chain builds a record whose fields are the chain's keys, over
shared `:draw` and `:chain` axes, with the record axis named `:param`:

```julia
X = TreeData(chain)                  # parameters only
X = TreeData(chain; extras=true)     # also `Extra` keys (e.g. `:lp`)
mean(X; dims=:draw)                  # ordinary reductions from here
```

Field names are `Symbol(key)`, so `VarName` optics survive textually (`y[1]`
and `y[2]` stay distinct); a `Parameter`/`Extra` pair collapsing to one name
throws. Scalar keys become `(draw, chain)` matrix fields. The `:draw`/`:chain`
coordinates come from the chain's index lookups, so `discard_initial`/thinning
offsets survive the crossing. Ingest copies once per key at construction
(`chain[k]` materializes; FlexiChains exposes no public zero-copy accessor) —
that copy is a construction cost, and reductions on the result never copy.

The crossing runs both ways, plus summaries:

```julia
FlexiChain(X)                        # rectangular numeric trees back to a chain
TreeData(summarystats(chain))        # a summary melted over its surviving dims
```

`FlexiChain(X)` accepts a dense leaf over exactly `:draw`/`:chain`/`:param`
(in any order) or a flat `:param` record — nested sub-trees unstack back to
array-valued keys — and refuses ragged, `Tuple`-backed, heterogeneous, and
scalar shapes by name. A fully-collapsed single stat (e.g. `mean(chain)`)
melts to a dense `:param` vector; anything else becomes a record over the
surviving `:draw`/`:chain`/`:stat` dims.

From there, everything else on this site applies unchanged:

```julia
post = TreeData(chain)

post.mu                                      # read one key back, still labelled
quantile(post, (lower = 0.025, median = 0.5, upper = 0.975); dims = (:draw, :chain))
TreeActualArray(post)                        # → ess / rhat, no copy
Tables.columns(post)                         # → any plotting layer, no DataFrame
```

Three things make the pairing more than cosmetic:

- **Pooled `(draw, chain)` reductions are one sorted pass over the whole bag** —
  exactly the quantity a pooled credible interval wants — and TreeArrays computes
  it identically whether the chains are one dense 3-D array or a *ragged* `:chain`
  axis of separately-allocated per-chain matrices. Chains that were sampled
  independently, possibly memory-mapped, never need `hcat`ing to be summarized
  together. See [Ragged data](@ref).
- **Array-valued keys keep their shape across the crossing.** Uniform cells nest
  as a sub-`TreeData` on the way in and unstack to array-valued keys on the way
  out; genuinely ragged per-draw data stays on the TreeArrays side as a
  [`TreeRaggedArray`](@ref) node, summarized together with the rest.
- **`TreeActualArray`** promotes a record axis to a real array dimension, so a
  record-of-per-parameter-matrices feeds `ess`/`rhat` with no copy and no parallel
  plain-array builder.

### Where this lives

In a **package extension**, not in TreeArrays' core: `TreeArraysFlexiChainsExt`,
behind a weak dependency on FlexiChains and loaded automatically when both
packages are loaded. A hard dependency on any modelling-ecosystem package is a
non-starter here — the core's dependencies are `Statistics` and `Tables`, and
that is intentional — exactly as the `NaNStatistics` extension does for
[`nanquantile`](@ref nan-safe-quantiles).
