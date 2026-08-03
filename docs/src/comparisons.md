```@meta
CurrentModule = TreeArrays
```

# Comparisons

TreeArrays sits next to two packages people reasonably ask about:
**DimensionalData.jl**, which also gives arrays named dimensions, and
**FlexiChains.jl**, which holds MCMC output. This page says what is genuinely
different, what is borrowed, and — for FlexiChains — what an integration would
look like.

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

!!! warning "This section is a sketch, not a shipped integration"
    TreeArrays has **no** FlexiChains dependency, extension, or tested bridge
    today, and the code below has not been run against FlexiChains. It describes
    the *shape* such an integration would take, so the design is written down
    somewhere. Check the FlexiChains documentation for its current API before
    relying on any spelling here.

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

### The natural bridge

The two type systems line up almost directly, because a chain object and a
`TreeData` describe the same logical thing — a `(draw, chain)` grid of keyed
values:

| FlexiChains | TreeArrays |
|---|---|
| the iteration dimension | an unlabelled `:draw` axis |
| the chain dimension | an unlabelled `:chain` axis |
| one scalar parameter key | a field of a [`TreeNamedTuple`](@ref) record, or a coordinate on a `:param` axis |
| one array-valued parameter key | a nested `TreeData` leaf with its own inner axes |
| keys whose per-draw values differ in size | a [`TreeRaggedArray`](@ref) node |

So a conversion would build a record whose fields are the chain's keys, over
shared `:draw` and `:chain` axes:

```julia
# SKETCH — not a supported API. Adapt the accessors to FlexiChains' actual surface.
function TreeArrays.TreeData(chain::FlexiChains.FlexiChain)
    fields = (; (Symbol(k) => _as_leaf(chain, k) for k in keys(chain))...)
    TreeData(:param => fields, :draw, :chain)
end

# a scalar key becomes a plain (draw, chain) matrix; an array-valued key becomes
# a nested TreeData carrying its own inner axes
_as_leaf(chain, k) = ...
```

Once that exists, everything on the rest of this site applies unchanged:

```julia
post = TreeData(chain)                       # zero-copy where the backing allows

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
- **FlexiChains' arbitrary-type values are what TreeArrays' record and ragged
  nodes are for.** A key whose draws are vectors of *differing* length is a
  ragged node, not a padding problem.
- **`TreeActualArray`** promotes a record axis to a real array dimension, so a
  record-of-per-parameter-matrices feeds `ess`/`rhat` with no copy and no parallel
  plain-array builder.

### Where this would live

Not in TreeArrays' core. A hard dependency on any modelling-ecosystem package is
a non-starter here — the core's dependencies are `Statistics` and `Tables`, and
that is intentional. The bridge belongs in a **package extension**, on whichever
side is willing to own it, exactly as the `NaNStatistics` extension does for
[`nanquantile`](@ref nan-safe-quantiles).

If you want this, [open an issue](https://github.com/nsiccha/TreeArrays.jl/issues)
— knowing which direction the conversion is actually needed in is the missing
input, not the implementation.
