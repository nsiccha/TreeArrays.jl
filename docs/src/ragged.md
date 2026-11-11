# Ragged data

Reduce observations on separate grids, pool compatible samples, and explicitly choose which coordinates contribute to an outer reduction.

## Separate grids

A ragged tree is an array of trees. Siblings share dimension names and layout, but may have different lengths and coordinates.

```@example ragged
using TreeArrays, Tables
labels = ["A", "B"]
leaves = [TreeData([2.0, 4.0, 6.0], :time => [0, 1, 2]),
          TreeData([10.0, 30.0], :time => [0, 2])]
X = TreeData(leaves, :subject => labels)
Tables.rowtable(X)
```

An inner reduction visits each leaf's own grid. The displayed peaks retain the outer subject labels.

```@example ragged
Tables.rowtable(mapslices(maximum, X; dims=:time))
```

Use `coords=true` when the kernel needs the grid; see [slice kernels](reductions.md#Slice-kernels). Reusing a single closed-over grid would give the wrong coordinates to some leaves.

## Coordinate alignment

Reducing subjects while retaining time needs a policy when the grids differ. `align=:union` uses every coordinate present in any leaf and only the contributors present there. `align=:intersect` uses only coordinates present in every leaf. Both match coordinate values, not positions; neither interpolates.

```@example ragged
union_mean = mean(X; dims=:subject, align=:union)
intersection_mean = mean(X; dims=:subject, align=:intersect)
(; union=Tables.rowtable(union_mean), intersection=Tables.rowtable(intersection_mean))
```

Report contributor counts alongside aligned summaries. A late coordinate based on fewer subjects has a different population from a coordinate everyone reached.

```@example ragged
contributors = mapslices(length, X; dims=:subject, align=:union)
Tables.rowtable(contributors)
```

Without `align`, a kept axis must agree across leaves in both length and labels. Explicit alignment is required even when a positional pairing would happen to fit the shapes.

## Pooled samples

Samples can be stored in separate chain matrices. Reduce both `:draw` and the outer `:chain` to pool all samples per parameter.

```@example ragged
chain_data = [reshape(Float64.(1:8), 4, 2), reshape(Float64.(9:20), 6, 2)]
chains = TreeData([TreeData(m, :draw, :parameter => [:a, :b]) for m in chain_data],
                  :chain => [1, 2])
intervals = quantile(chains, (lower=0.1, median=0.5, upper=0.9);
                     dims=(:draw, :chain))
Tables.rowtable(intervals)
```

This pooling path requires array-backed leaves and matching kept axes. If a kept axis differs, choose `:union` or `:intersect` by value. For a record, select the numeric field before pooling. All named outer axes must participate in a pooled outer-plus-inner reduction; keeping another outer axis is unsupported.

## Metadata

Do not encode varying sibling attributes solely as fixed metadata inside leaves. In the conformable ragged table path, a fixed coordinate can be taken from the first sibling. Put a varying attribute on an outer axis or in a record field so that it is represented as data.

For an `AbstractArray` consumer, the [array-view alignment policies](tables.md#Numeric-arrays) select positions and have a different purpose from union/intersection reductions.

The [concentration example](trajectories.md) keeps separate subject grids while reducing draws and calculating exposure.
