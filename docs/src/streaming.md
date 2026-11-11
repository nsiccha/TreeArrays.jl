# Streaming and caches

Keep per-draw statistics without retaining every trajectory, then reopen a tree with mapped numeric backing.

## Per-draw summaries

Generate a trajectory, calculate the requested statistics, and retain only their values. A named sweep preserves draw identifiers, which are needed for later quantiles or joint summaries.

```@example streaming
using TreeArrays, Tables
trajectory(draw) = [Float64(draw) * exp(-t / 2) for t in 0:4]
compact = map(TreeDim(:draw, 1:4)) do draw
    y = trajectory(draw)
    TreeData(:stat => (peak=maximum(y), total=sum(y)))
end
Tables.rowtable(compact)
```

The generator runs once per draw in this loop. The tree retains the returned statistics; the local `y` is not part of the result. Returning a view of `y` would retain its backing, so a summary intended to release the trajectory should return independent values or arrays.

```@example streaming
Tables.rowtable(quantile(compact.peak, (lower=0.1, median=0.5, upper=0.9); dims=:draw))
```

## Packed columns

For a fixed schema, collect each statistic into its own column. This layout also gives a memory-mapped cache one numeric leaf per statistic, instead of a separate leaf for every draw-statistic pair. Select the column types to match the kernel's outputs.

```@example streaming
function packed_statistics(generator, draw_labels)
    peak = Vector{Float64}(undef, length(draw_labels))
    total = similar(peak)
    for (i, draw) in enumerate(draw_labels)
        y = generator(draw)
        peak[i] = maximum(y)
        total[i] = sum(y)
    end
    draw_axis = TreeDim(:draw, draw_labels)
    fields = map(v -> TreeData(v, draw_axis), (; peak, total))
    TreeData(:stat => fields, draw_axis)
end
packed = packed_statistics(trajectory, collect(1:4))
Tables.rowtable(packed)
```

Each column is private to the calculation until the record is assembled. This function keeps the per-draw summaries available; computing an exact quantile later still needs those retained sample values.

## Mapped storage

Loading DynamicObjects activates the cache codec. Save the tree and reopen it as `TreeData`; the loaded backing types can differ from the original array types.

```@example streaming
using DynamicObjects
cache_check = mktempdir() do directory
    path = joinpath(directory, "statistics.mmap")
    DynamicObjects.save(Val(:mmap), path, packed)
    restored = DynamicObjects.load(Val(:mmap), path, TreeData)
    (; same_values=Tables.rowtable(restored) == Tables.rowtable(packed),
       draw_labels=coords(restored, :draw),
       peaks=collect(parent(restored.peak)))
end
@assert cache_check.same_values
cache_check
```

Numeric array leaves reopen as mapped arrays; scalar terminals reopen as scalar values. The structural metadata carries dimension names, coordinates, record fields, and reduced dimensions. A codec round trip is storage, not automatic cache invalidation: the caller must choose when an existing file still describes the current computation.

## Computation keys

A DynamicObjects owner can include the generator and an implementation version in a cached property's identity. Keep the callable an explicit argument when its configuration changes the result.

```@example streaming
@dynamicstruct struct DrawSummary
    cache_directory
    draw_labels
    __cache_base__ = cache_directory
    @mmap v"1" result(generator)::TreeData = packed_statistics(generator, draw_labels)
end
owner_check = mktempdir() do directory
    job = DrawSummary(directory, collect(1:4))
    first_result = job.result(trajectory)
    second_result = DrawSummary(directory, collect(1:4)).result(trajectory)
    (; same_values=Tables.rowtable(first_result) == Tables.rowtable(second_result),
       columns=Tables.columnnames(Tables.columns(second_result)))
end
@assert owner_check.same_values
owner_check
```

Keep mapped results alive while downstream views use their backing. Numeric arrays must use the codec's supported storage types; arbitrary object arrays are not portable mapped leaves. Avoid requesting the original concrete `TreeData` type at load time if its backing type is no longer correct.

The [concentration example](trajectories.md#Exposure) computes an exposure statistic within each draw before reducing the draw axis. The same order can be used in a generator that retains exposures alone.
