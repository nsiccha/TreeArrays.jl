```@meta
CurrentModule = TreeArrays
```

# Getting started

## Installation

TreeArrays is not registered. Add it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/TreeArrays.jl")
```

Its only hard dependencies are `Statistics` and `Tables`. `NaNStatistics` is a
weak dependency that lights up [`nanquantile`](@ref nan-safe-quantiles) when you
load it.

## The mental model

A [`TreeData`](@ref) is a **`parent`** (the backing storage) plus **`dims`** (the
named axes describing it). That is the whole type. What `parent` *is* decides
which of four shapes you have — and these are dispatch-only aliases, so you
always construct with `TreeData(...)`:

| Alias | `parent` is… | Meaning |
|---|---|---|
| [`TreeArray`](@ref) | an `AbstractArray` | a dense numeric leaf |
| [`TreeNamedTuple`](@ref) | a `NamedTuple` | a **record** — named fields form an axis |
| [`TreeTuple`](@ref) | a `Tuple` | a positional record |
| [`TreeRaggedArray`](@ref) | an `AbstractArray{<:TreeData}` | a **ragged** nesting of sub-trees |

A `TreeData` is deliberately **not** an `AbstractArray` subtype. Only the
array-backed leaf forwards `size`/`getindex`/`iterate` to its parent. See
[Feeding a numeric-array API](@ref actual-array) for how to cross that boundary
when a function insists on an array.

## Naming your axes

```@example gs
using TreeArrays, Random
Random.seed!(1)

X = TreeData(randn(500, 3, 5), :draw, :subject, :time => [0.0, 1.0, 2.0, 4.0, 8.0])
TreeArrays.dims(X)
```

The positional arguments after the backing array name its axes, in order. Three
things are happening there:

- `:draw` and `:subject` are **unlabelled axes** — real dimensions with no
  coordinates attached. Their `values` is `missing`.
- `:time => [...]` is a **labelled axis**: the coordinates travel with the data.
- Keyword arguments become **fixed coordinates** — a single position, not an
  axis:

```@example gs
D = TreeData(randn(500, 3), :draw, :subject; dose = 20, arm = :treatment)
TreeArrays.dims(D)
```

`dose = 20` is stored **once**, as a label. This is the constant-column smear
(`df[!, :dose] .= 20`) that TreeArrays exists to retire — it costs O(1), not
O(rows).

!!! warning "`TreeData(:name)` with no backing array is a *builder*"
    `TreeData(dims::Symbol...)` curries: it returns a closure
    `(x, vals...) -> TreeData(x, dims...)`, which is what makes the `map` form in
    [Ragged data](@ref) work. So `TreeData(:time)` is a constructor waiting for
    data, not a value. Don't pass it where you meant a constructed tree.

## Reading the labels back

[`coords`](@ref) is the supported way to get an axis's coordinates:

```@example gs
coords(X, :time)
```

Labels come back **exactly as provided** — a `Tuple` stays a `Tuple`, a `Vector`
stays a `Vector`, a range stays a range. The same convention holds for
`quantile`'s levels and `selectdim`'s.

!!! danger "Do not reach for `collect(dim)`"
    A `TreeDim` has `iterate`/`length`, so `collect` *runs* and looks like the
    answer. It isn't: the eltype is `Any`, and for the two dim kinds that have
    **no** coordinates it invents one — `Any[missing]` for an unlabelled axis,
    `Any[nothing]` for a reduced ghost. `coords` errors by name for both, and
    the message names the spelling that would have worked.

## Your first reduction

Every reduction names the axes it collapses and leaves the rest alone:

```@example gs
using Statistics
m = mean(X; dims = :draw)
TreeArrays.dims(m)
```

The `:draw` axis is not dropped — it is kept as an **aggregated ghost**
(`values === nothing`), so provenance survives the reduction and the result still
knows what it is. `:subject` and `:time` are untouched, labels included.

`dims = nothing` (the default) reduces the whole tree to a scalar:

```@example gs
mean(X; dims = nothing)
```

## Names are checked — a typo throws

```@example gs
try
    mean(X; dims = :drwa)
catch err
    println(err)
end
```

`dims =` is **foundALL**: every name must resolve somewhere in the tree. This is
checked from the *type*, so a correct call costs nothing at runtime; only on a
jagged tree does it fall back to an instance walk rather than let a typo through.

The rule behind this shows up everywhere in TreeArrays: **never silently return a
plausible-looking value**. An earlier design returned `missing` for a dim that
resolved nowhere, and it was retired precisely because a typo was then
indistinguishable from a legitimately absent axis.

## Small anatomy of the result

The whole tree renders itself, dims table included:

```@example gs
Y = TreeData(reshape(1.0:12.0, 4, 3), :draw, :time => [0.0, 1.0, 2.0])
```

```@example gs
mean(Y; dims = :draw)
```

Display is always **bounded** and every elision is marked — `string(td)` is a
preview, never a serialization. If you want the data, use
[`Tables.columns`](@ref tables) or `parent(td)`.

## Where to go next

- [Reductions](@ref) — `mapslices`, `quantile`, `coords = true`, `@kernel`, chaining.
- [Ragged data](@ref) — per-subject series of differing length.
- [Tables & plotting](@ref tables) — melt long, pivot wide, feed a plotting layer.
- [Comparisons](@ref) — how this relates to DimensionalData.jl and FlexiChains.jl.
