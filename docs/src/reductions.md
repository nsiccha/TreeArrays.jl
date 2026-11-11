```@meta
CurrentModule = TreeArrays
```

# Reductions

Reduce dimensions by name, choose the result's structure, and write kernels that use each slice's coordinates.

## Named dimensions

A reduction removes variation along the requested dimensions and keeps the other axes. Name several dimensions together to pool their values in one calculation.

```@example reductions
using TreeArrays, Tables, Statistics
X = TreeData(reshape(Float64.(1:24), 4, 2, 3),
             :draw, :chain, :time => [0.0, 1.0, 2.0])
pooled = mean(X; dims=(:draw, :chain))
Tables.rowtable(pooled)
```

Pooling samples is different from reducing summaries in sequence. For example, a quantile of chain quantiles is generally not the pooled quantile. Supply every sample axis in the same `dims` argument when pooling is intended.

| Operation | Result per slice |
| --- | --- |
| [`mean`](@ref), [`sum`](@ref) | Mean or sum |
| [`std`](@ref), [`var`](@ref) | Standard deviation or variance; `corrected=true` by default |
| [`quantile`](@ref) | Requested quantile levels |
| [`hdi`](@ref) | Shortest sample interval containing at least the requested mass |
| [`mapslices`](@ref) | A scalar or tree returned by the supplied function |

## Quantile layouts

Choose the output structure according to the next operation. A named tuple produces a record whose fields become table columns.

```@example reductions
wide = quantile(X, (lower=0.1, median=0.5, upper=0.9); dims=(:draw, :chain))
Tables.rowtable(wide)
```

A `TreeDim` produces an axis with numeric probability coordinates. This long representation keeps levels available for selection and further tree operations.

```@example reductions
long = quantile(X, TreeDim(:probability, (0.1, 0.5, 0.9)); dims=(:draw, :chain))
Tables.rowtable(long)
```

A name paired with a named tuple produces an axis with named coordinates. `TreeTable` can pivot that axis at the presentation boundary.

```@example reductions
named = quantile(X, :band => (lower=0.1, median=0.5, upper=0.9);
                 dims=(:draw, :chain))
Tables.rowtable(TreeTable(named; wide=:band))
```

All forms calculate the requested levels together. See [table views](tables.md#Tables) for the long/wide boundary.

## Missing numeric values

`quantile` rejects `NaN`. Loading NaNStatistics activates `nanquantile`, which excludes NaNs independently in each slice and returns NaN for an all-NaN slice. It does not treat Julia's `missing` as a numeric value.

```@example reductions
using NaNStatistics: nanquantile
incomplete = TreeData([1.0 NaN; NaN NaN; 3.0 NaN], :draw, :time => [0, 1])
Tables.rowtable(nanquantile(incomplete, (median=0.5,); dims=:draw))
```

## Sample intervals

`hdi` returns observed sample endpoints, without interpolation. Its interval covers at least the requested sample mass; it does not guarantee coverage of a population distribution. For multimodal samples a single interval may span a low-density gap.

```@example reductions
Tables.rowtable(hdi(X; dims=(:draw, :chain), mass=0.8))
```

## Slice kernels

`mapslices` passes each reduced slice to a function. Returning a scalar collapses the dimension. Returning `TreeData` introduces an output axis or record. A plain named tuple must be wrapped as a record so that the returned fields have a named axis.

```@example reductions
spread = mapslices(X; dims=(:draw, :chain)) do y
    TreeData(:stat => (smallest=minimum(y), largest=maximum(y)))
end
Tables.rowtable(spread)
```

For a single labelled dimension, `coords=true` passes its coordinates as a second argument. This supports integrals and event times on irregular grids. `@kernel` gives a reusable function an input dimension and a record output dimension:

```@example reductions
@kernel (:time => :stat) function peak(y, t)
    i = argmax(y)
    (height=y[i], time=t[i])
end
signal = TreeData([1.0, 3.0, 2.0], :time => [0.0, 0.5, 2.0])
Tables.rowtable(peak(signal))
```

The plain-array method still returns the named tuple. The generated tree method wraps that result as the `:stat` record and supplies the labelled time coordinates. Use `mapslices` directly when the consumed dimensions vary between calls.

## Execution

Built-in reducers enable threading by default. `mapslices` and generated `@kernel` methods opt in with `threads=true`. A threaded kernel must be safe to call concurrently and must not mutate input slices. Sorting reducers copy their slices into scratch storage; parallel tasks may each retain a buffer. Use `threads=false` to keep the calculation on the calling task.

The [concentration example](trajectories.md) reduces draws for pointwise intervals and time for per-draw exposure, then summarizes the exposures.
