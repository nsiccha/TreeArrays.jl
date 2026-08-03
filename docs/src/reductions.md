```@meta
CurrentModule = TreeArrays
```

# Reductions

Every reduction in TreeArrays follows one shape: **name the axes to collapse,
keep everything else**. The result is still a `TreeData`, so reductions compose
by nesting.

```@example red
using TreeArrays, Statistics, Random
Random.seed!(1)

times = [0.0, 1.0, 2.0, 4.0, 8.0]
X = TreeData(randn(500, 3, 5), :draw, :subject, :time => times)
TreeArrays.dims(X)
```

## `mapslices` — the workhorse

[`mapslices`](@ref)`(f, X; dims)` reduces only the named `dims`, threads
every leftover index recursively, and stays a `TreeData`. Your kernel `f`
receives the **slice along the reduced axis**.

A **scalar-returning** kernel collapses the axis:

```@example red
peak = mapslices(maximum, X; dims = :time)
TreeArrays.dims(peak)
```

A **`TreeData`-returning** kernel introduces a new output axis instead — this is
how you get several statistics out of one pass:

```@example red
stats = mapslices(X; dims = :time) do y
    TreeData(:stat => (; peak = maximum(y), trough = minimum(y), baseline = y[1]))
end
TreeArrays.dims(stats)
```

The `:stat` record becomes a real axis, and the fields keep their names all the
way to the table boundary.

## `mean`, `sum`, `var`, `std`

```@example red
TreeArrays.dims(mean(X; dims = :subject))
```

```@example red
sum(X; dims = nothing)     # the whole tree
```

`var`/`std` take `corrected` and reduce the named axes as one pooled bag, the
same way `mean` does.

## `quantile` — all levels in one pass

[`quantile`](@ref) is the output-axis reducer. You hand it the *name and levels
of the output axis*, and it computes every level in **one sorted pass** over a
shared scratch buffer.

```@example red
r = quantile(X, TreeDim(:band, (0.1, 0.5, 0.9)); dims = :draw)
TreeArrays.dims(r)
```

Never loop `quantile` once per level — pass the level collection and get one pass
plus one output axis.

Levels are kept exactly as given: a `Tuple` of levels gives a `Tuple`-backed
leaf, a `Vector` gives a `Vector`, and a **scalar** `p` mirrors Base — one number
per slice, recorded as a fixed coordinate rather than a length-1 axis:

```@example red
TreeArrays.dims(quantile(X, TreeDim(:median, 0.5); dims = :draw))
```

### Three spellings, two orientations

The same reduction can hand you a long axis or a wide record, and that choice is
yours to make at reduction time:

```@example red
# 1. TreeDim  → a labelled AXIS (melts long)
quantile(X, TreeDim(:band, (0.1, 0.5, 0.9)); dims = :draw) |> TreeArrays.dims
```

```@example red
# 2. NamedTuple → a RECORD (melts wide, feeds `bands = [:lower => :upper]` directly)
w = quantile(X, (lower = 0.1, median = 0.5, upper = 0.9); dims = :draw)
TreeArrays.dims(w)
```

```@example red
# 3. name => NamedTuple → a Symbol-labelled AXIS (long, but pivotable — see the Tables page)
quantile(X, :band => (lower = 0.1, median = 0.5, upper = 0.9); dims = :draw) |> TreeArrays.dims
```

Reach for the **record** form when you always want the bands as columns; reach
for the **axis** form when the same reduction feeds both a tidy long consumer and
a ribbon plot, or when you still need to reduce *over* the band axis.

### [NaN-safe quantiles](@id nan-safe-quantiles)

`quantile`/`quantile!` delegate to Base, so they **throw on `NaN`**. Load
`NaNStatistics` and the `nanquantile` extension takes over — same call shape,
same output axis, NaNs dropped per slice:

```@example red
using NaNStatistics
Xn = TreeData([1.0, NaN, 2.0, 3.0], :draw)
nanquantile(Xn, TreeDim(:band, (0.25, 0.5, 0.75)); dims = :draw)
```

An all-`NaN` slice yields `NaN` at every level rather than throwing.

## `coords = true` — let the kernel see the axis

By default a kernel sees the **data slice only**. Positional tricks (`y[1]`,
`y .- y[1]`) get you a long way, but AUC and `tmax` need the actual grid.
`coords = true` hands it over as a second argument:

```@example red
trapz(t, y) = sum((t[i+1] - t[i]) * (y[i+1] + y[i]) / 2 for i in 1:length(t)-1)

nca = mapslices(X; dims = :time, coords = true) do y, t
    TreeData(:stat => (; cmax = maximum(y), tmax = t[argmax(y)], auc = trapz(t, y)))
end
TreeArrays.dims(nca)
```

The binding happens **per node**, which is the whole point: on a ragged tree every
sub-tree's kernel call sees *its own* grid. That is why "close over a shared
constant vector" was never an answer for ragged data — the coordinates differ per
sub-tree.

- **One** reduced axis → the bare coordinate vector, `f(y, t)`.
- **Several** → one vector per axis, in slice-dim order, `f(s, (draws, chains))`,
  because the slice genuinely is N-dimensional there.
- Reducing a ragged tree's **outer** axis hands the kernel that axis's labels.

Two shapes throw by name instead of answering with something plausible: an
**unlabelled** axis has no coordinates to give, and a **pooled straddle**
concatenates every leaf's slice into one bag, so no single axis's coordinates
line up with it.

## `@kernel` — a dimension-aware kernel from plain math

Write the kernel once as ordinary array math, annotate it with its
`:reduces => :into` signature, and get a `TreeData` method for free:

```@example red
@kernel (:time => :stat) function summarize(y, t)
    (; cmax = maximum(y), tmax = t[argmax(y)], auc = trapz(t, y))
end

summarize([1.0, 3.0, 2.0], [0.0, 1.0, 2.0])   # still a plain-array function
```

```@example red
TreeArrays.dims(summarize(X))                 # …and now also a TreeData reducer
```

The **arity of your literal argument list** decides whether the kernel gets
coordinates: a 2-argument kernel gets them, a 1-argument one does not. The macro
can read that argument list, so the opt-in is automatic and no existing kernel
changes behaviour.

!!! note "Why `mapslices` doesn't sniff arity too"
    At the `mapslices` boundary a probe would be actively dangerous:
    `hasmethod(maximum, (y, t))` is **true**, because `maximum(f, itr)` exists.
    Sniffing would answer `mapslices(maximum, X; dims = :time)` by treating your
    data slice as a predicate. Hence: explicit kwarg for a bare kernel, inferred
    for `@kernel`.

There is also a **post-hoc** form, `@kernel (:time => :stat) f`, which adds the
`TreeData` method to an existing `f`. It has no argument list to read, so it is
1-argument by default with an explicit opt-in:
`@kernel (:time => :stat) coords=true nca`.

Parameterized reducers — `mean`, `sum`, `quantile`, where the axis and output are
chosen per call — don't fit the fixed-signature shape. Use their `dims =` methods.

## Chaining across axes

Reduce different axes in sequence by nesting the calls. Each takes the previous
result, and each keeps every other axis and label. This is the canonical PKPD
summary shape, `reduce(time) ∘ reduce(subject) ∘ reduce(draw)`:

```@example red
chained = quantile(
    quantile(
        mapslices(maximum, X; dims = :time),
        TreeDim(:percentile, (0.05, 0.5, 0.95)); dims = :subject),
    TreeDim(:ribbon, (0.025, 0.5, 0.975)); dims = :draw)
```

The result stays a tree of arrays the whole way down. No intermediate flat frame
is ever built — that is the difference from a repeated `combine`/`groupby`
pipeline, where every stage materializes a table whose columns are mostly
repeated labels.

## `selectdim` — the selection dual

The reductions **collapse** a named axis. [`selectdim`](@ref) **restricts** one,
leaving every other axis intact:

```@example red
P = TreeData(randn(500, 4), :draw, :param => ("beta[1]", "beta[2]", "sigma", "lp__"))

sel = selectdim(P, :param => r"^beta")
coords(sel, :param)
```

```@example red
parent(sel) isa SubArray   # no data was copied
```

Five selector spellings, all through the same call:

```julia
selectdim(X, :param => r"^unit_")          # Regex — keep labels it `contains`-matches
selectdim(X, :param => startswith("beta")) # a `label -> Bool` predicate
selectdim(X, :param => boolmask)           # an explicit Bool mask
selectdim(X, :param => [8, 1, 2])          # integer indices — subset and/or reorder
selectdim(X, :param, sel)                  # 3-arg form, mirroring Base.selectdim
```

**No data is copied**: the backing array is sliced with a `view`, so a subset of a
`(draw × chain × 600_000)` matrix never materializes — only the axis's own tiny
label vector is subset. Positional selectors (a mask, integer indices) also work
on an *unlabelled* axis; a Regex or predicate on one throws, because there are no
labels to match.

## Type stability — the function barrier is the lever

`mapslices` specializes on `typeof(f)`, so passing the kernel as a concrete
function **value** monomorphizes the hot inner loop. This is the single biggest
win when replacing a DataFrames pipeline:

```julia
mapslices(signed_max_abs, X; dims = :time)   # ✅ a named function ⇒ specialized
```

```julia
mapslices(X; dims = :time) do y              # ❌ branches on a runtime String
    qoi == "max" ? maximum(y) : minimum(y)   #    inside the slice loop ⇒ dispatch per element
end
```

Select the concrete kernel **once**, outside, then pass it. This is exactly what
`combine(groupby(df, keys), :col => (v -> f(v)))` cannot do: an anonymous closure
across the split-apply-combine boundary returns `Any`-typed columns and nothing
infers.

## Scenario sweeps

`map` over an `Iterators.product` of `TreeDim`s builds a cartesian sweep whose
axes are the swept dims, and whose cells are whatever the function returns:

```@example red
ds = (TreeDim(:dose, (20, 200)), TreeDim(:arm, ("a", "b")))

sweep = map(Iterators.product(ds...)) do (dose, arm)
    dose * length(arm)
end
```

The function receives the **coordinate tuple** in dim order — destructure it. It
is arity-generic (regression-pinned at 1, 3, 6, 8 and 10 dims), label types may be
heterogeneous, and a returned `TreeData` stays a leaf, so a sweep of reduced
bands composes with further reductions like any other tree.

A **scalar** dim is a fixed position and contributes zero array axes; pass a
1-tuple (`TreeDim(:schedule, ("s",))`) to sweep one value as a real length-1
axis. An **unlabelled** dim has nothing to sweep and errors by name.

!!! tip "Splat a `Tuple`, not a generator"
    Splatting a *runtime-length* container of dims is inference-opaque — as any
    runtime-length splat is in Julia, not a TreeArrays effect. It still works; to
    also infer, splat a `Tuple` (`Tuple(ds)`) or cross a `Tuple`-typed function
    barrier.

## Retiring `combine` / `groupby` / `vcat`

A `combine(groupby(df, keys), :val => f)` is "reduce `val` over the axis that
`keys` does *not* name". In TreeArrays that axis is a real dim:

| DataFrames pattern | TreeArrays |
|---|---|
| `combine(groupby(df, everything_except(:x)), :v => mean)` | `mean(X; dims = :x)` |
| `combine(groupby(df, …), :v => (v -> quantile(v, ps)))` | `quantile(X, TreeDim(:q, ps); dims = :x)` |
| repeated `combine`/`groupby` for a nested summary | **chained** reductions — no intermediate frame |
| `reduce(vcat, per_block_frames)` to pool blocks | an outer axis (`:dose`, `:scenario`) — pooling is *structure*, not a copy |
| a constant column `df[!, :dose] .= D` | a fixed dim `dose = D` — stored once, O(1) |
