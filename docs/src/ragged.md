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

rag = TreeData(map((v, t) -> TreeData(v, :time => t), values, times), :subject => ["s1", "s2", "s3"])
```

The closure builds one leaf per subject, so each subject ends up carrying
**its own** `:time` coordinates.

Nothing was copied. If your series already live as columns of a shared matrix,
build the leaves from `view`s and the ragged tree is pure structure over that one
allocation:

```julia
subjects = TreeData(map(s -> TreeData(view(mat, :, idx[s]), :draw, :time), 1:n), :subject)
```

!!! note "Siblings share a *layout*, not a *shape* or a *type*"
    Differing lengths are exactly what raggedness is, and differing *storage*
    is fine too: one subject's `:time` coordinates as a 4-`Tuple` and another's
    as a 3-`Tuple`, a `view` beside a `Vector`, or a `TreeData[]` filled in a
    loop all widen the container's eltype to something non-concrete, and the
    tree still reduces and melts. What must agree is the layout — dim names and
    kinds, element types, nesting — and siblings that differ there are refused
    by name. `Vector` coordinates (as above) keep every leaf one concrete type,
    which keeps the per-row melt type-stable.

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
recurses. Nothing is materialized on the way. It pools position `k` of every
leaf, so the leaves must agree on every kept axis — same length **and** same
labels. Leaves that do not are refused by name, unless you say how to match
them ([`align`](@ref ragged-align) below).

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
  shares `:param`. A kept axis that is itself ragged is refused, unless `align`
  says how to match it ([next section](@ref ragged-align)).
- **Leaves must be array-backed.** Pooling across a record, tuple, or
  doubly-ragged leaf is not implemented — select one field first (`X.field`
  maps over the leaves), or reduce the inner dims first.

A purely *inner* multi-dim reduce (`dims = (:time, :channel)`, neither of them the
outer axis) never touches the pooling path; it just reduces inside each leaf.

## [When the kept axis differs: `align`](@id ragged-align)

Participants in a behavioural experiment rarely all complete the same number of
trials. Every participant then has its own `:trial` axis:

```@example rag
trials = [6, 6, 4, 5]
participants = TreeData([TreeData(randn(200, 2, n), :draw, :chain, :trial => 1:n) for n in trials],
                        :participant => 1:4)
nothing # hide
```

A band per trial across participants pools `:draw`, `:chain` and `:participant`
while **keeping** `:trial`, and `:trial` differs between the leaves. There is
no single right answer to "the band at trial 6" here: only two of the four
participants reached it. So the reduction refuses unless you say how to match
the trials, with `align`:

```@example rag
band = quantile(participants, (lower = 0.025, median = 0.5, upper = 0.975);
                dims = (:draw, :chain, :participant), align = :union)
```

- `align = :union` — every trial any participant has; at each, pool the
  participants that have it. Later trials rest on fewer participants.
- `align = :intersect` — only the trials every participant has (here 1–4).

The same keyword works when only the outer axis is reduced, for example the
posterior of the mean across participants, trial by trial:

```@example rag
mean_traj = mean(participants; dims = :participant, align = :union)
quantile(mean_traj, (lower = 0.025, upper = 0.975); dims = (:draw, :chain))
```

How many participants went into each trial is one more reduction, counted on a
tree that has one trajectory per participant:

```@example rag
per_participant = mean(participants; dims = (:draw, :chain))
mapslices(length, per_participant; dims = :participant, align = :union)
```

Labels are matched by **value**, never by position. So subjects sampled at
different times pool only at the times they share, an unlabelled axis that
would need aligning throws (give it coordinates, `:trial => 1:n`), and so does
a label repeated within one leaf. The aligned axis is sorted when every leaf's
labels are ascending numbers; otherwise labels appear in the order first met.
When the leaves already agree, `align` changes nothing. Every built-in reducer
takes it, and so does `mapslices`.

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
