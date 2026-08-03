```@meta
CurrentModule = TreeArrays
```

# Ragged data

Real per-subject data is rarely rectangular. Subject 1 was sampled four times,
subject 2 three times, subject 3 five times; the dose histories have a *different*
irregular structure again. The usual escapes — pad to the longest, or flatten to
a long frame and carry the subject id on every row — both cost you something you
did not want to pay.

A [`TreeRaggedArray`](@ref) is an outer axis of **sub-trees with different
shapes**. It is a normal `TreeData` whose `parent` happens to be an array of
`TreeData`s.

## Building one

```@example rag
using TreeArrays, Statistics, Random
Random.seed!(1)

times = [[0.5, 1.0, 2.0, 4.0], [0.5, 1.5, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
values = [randn(length(t)) for t in times]

rag = TreeData(map(TreeData(:time), values, times), :subject => ["s1", "s2", "s3"])
```

This is where the currying footgun from [Getting started](@ref) becomes the
feature: `TreeData(:time)` is a *builder*, so `map` applies it pairwise and each
subject ends up carrying **its own** `:time` coordinates.

Nothing was copied. If your series already live as columns of a shared matrix,
build the leaves from `view`s and the ragged tree is pure structure over that one
allocation:

```julia
subjects = TreeData(map(s -> TreeData(view(mat, :, idx[s]), :draw, :time), 1:n), :subject)
```

!!! warning "Siblings must share a *type*, not a *shape*"
    Differing lengths are exactly what raggedness is. What breaks is differing
    *types*: give one subject's `:time` coordinates as a 4-`Tuple` and another's
    as a 3-`Tuple` and the container's eltype widens to something non-concrete,
    which has no static schema to serve. Use `Vector` coordinates (as above) —
    every leaf then shares one concrete type and lengths are free to differ.

## Reducing it

The same call you would write for rectangular data:

```@example rag
peaks = mapslices(maximum, rag; dims = :time)
```

Each sub-tree is reduced over **its own** `:time` axis. After that collapse the
per-subject scalars are regular again, so the subsequent `:subject` and `:draw`
reductions are ordinary dense reductions.

Reducing the **outer** axis is not forbidden either — it pushes the axis down to
the leaves as a streaming gather (pure index arithmetic on a gathered slice) and
recurses. Nothing is materialized on the way.

A coordinate-aware kernel works here too, and this is the case that motivates
`coords = true` in the first place — the grid genuinely differs per subject, so
no closed-over constant could have supplied it:

```@example rag
trapz(t, y) = sum((t[i+1] - t[i]) * (y[i+1] + y[i]) / 2 for i in 1:length(t)-1)

auc = mapslices(rag; dims = :time, coords = true) do y, t
    trapz(t, y)
end
```

## Pooling an outer axis together with an inner one

Hold per-chain draws as **self-contained** `(draw, param)` matrices — each its own
memory map, if you like — and collect them lazily as a ragged `:chain` axis. They
are never `hcat`ed into one block. A pooled credible interval then wants a
quantile over *all* draws × *all* chains, per parameter. Name **both** axes:

```@example rag
mats = [randn(400, 3) for _ in 1:4]
chains = TreeData([TreeData(m, :draw, :param => (:a, :b, :c)) for m in mats], :chain)

pooled = quantile(chains, (lower = 0.1, median = 0.5, upper = 0.9); dims = (:draw, :chain))
```

This **pools** `:draw` and `:chain` into one bag per parameter, and it is
**byte-identical** to the same reduction on the dense
`TreeData(arr3d, :draw, :param => names, :chain)`. That is worth stating plainly:
a pooled reduce is *one sorted pass over the whole `(draw × chain)` bag*, not the
composition of two reductions — it is not `quantile`-of-per-chain-`quantile`s.

It streams: the per-chain backing arrays are referenced, never copied or stacked,
and peak memory is about one parameter's pooled slice. An eager
`pooled = reduce(hcat, per_chain_matrices)` can be deleted outright.

Three requirements, each of which errors by name rather than pooling wrongly:

- **Every named outer axis must be reduced.** Pooling one outer axis while
  *keeping* another (a 2-D grid of ragged cells) is not implemented.
- **The kept inner axes must be conformable across leaves** — here every chain
  shares `:param`. A kept axis that is itself ragged is refused.
- **Leaves must be array-backed.** Pooling across a record, tuple, or
  doubly-ragged leaf is not implemented — reduce the inner dims first.

A purely *inner* multi-dim reduce (`dims = (:time, :channel)`, neither of them the
outer axis) never touches the pooling path; it just reduces inside each leaf.

## Derived quantities per chain, before the pool

There is **no `map`-over-leaves API**, and `map(f, ragged_tree)` silently drops
the outer axis — the ragged tree forwards `iterate` to its parent, so you get a
bare `Vector` back. Iterate the *leaves* and re-wrap yourself:

```@example rag
derived = TreeData(
    [TreeData(exp.(parent(leaf)), :draw, :derived => (:a, :b, :c)) for leaf in parent(chains)],
    :chain)

quantile(derived, (lower = 0.1, median = 0.5, upper = 0.9); dims = (:draw, :chain))
```

The comprehension is the canonical pattern, and it composes with the pooled
straddle above.

## A ragged tree is a table

You do **not** have to reduce a ragged tree to rectangularity before plotting it.
It melts long directly, because each row's coordinate is read from the position it
occupies in its **own** sub-tree:

```@example rag
using Tables
Tables.schema(rag)
```

```@example rag
length(Tables.columns(rag).value)   # sum(length, times) — not 5 × 3
```

Row count is `sum(length, leaves)`, not a product of axis extents. Each sibling's
rows are contiguous (the outer axis varies slowest), so every column is the
per-leaf concatenation. Row *order* is not part of the Tables.jl contract; don't
depend on it across shapes.

This is what retires the hand-rolled melt: the `reduce(vcat, values)` plus
`fill(label, n)` pair that smears a per-subject scalar across its own rows is
exactly the metadata smear TreeArrays exists to avoid — and it tends to get
reintroduced by hand at precisely the boundary TreeArrays owns.

Two things worth knowing about the long melt of a ragged tree:

- **Per-subject fixed dims come along correctly** — a `dose = 20` beside a
  `dose = 200` reports each subject's own value whenever the siblings differ in
  shape or in coordinates. Siblings that agree on **both** and differ *only* in a
  fixed value are not detected, and report the representative's value for every
  row. This is a deliberate permanent boundary (detecting it costs O(rows)):
  **express a value that legitimately varies per sibling as an axis coordinate**,
  not as a per-element fixed dim.
- **It is still lazy.** A coordinate column across a ragged boundary is a
  view-with-a-rule like any other, just with an offset table instead of a stride.
  `@allocated` stays flat across 100× rows.

The one exception is `wide =` — a pivot needs one level set and one column
length, and a ragged axis has neither, so `TreeTable(rag; wide = …)` throws and
says to drop the `wide =`. See [Tables & plotting](@ref tables).
