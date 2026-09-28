```@meta
CurrentModule = TreeArrays
```

# Streaming statistics and disk caches

Generate one trajectory, compute every requested statistic from it, then move
to the next draw. The existing named `map` collects the compact records;
it does not collect a draw-by-time trajectory matrix.

```@example streaming
using TreeArrays

trajectory(draw, ntime) = [Float64(draw) + j / ntime for j in 1:ntime]
above(threshold) = y -> count(v -> v > threshold, y)
statistics = (; total=sum, above=above(2.5))

compact = map(TreeDim(:draw, [1, 2, 3, 4])) do draw
    y = trajectory(draw, 1024)
    TreeData(:stat => map(f -> fill(f(y)), statistics))
end

propertynames(compact)
```

Each `fill(f(y))` creates a fresh zero-dimensional numeric array holding one
statistic. These independent arrays can retain mapped backing after a disk
reopen. No input, statistic argument, retained view, or earlier result is
overwritten. A statistic returning an array can return its own fresh array
directly. Existing kernels that return scalar records keep their scalar
semantics; scalar terminals reopen as values.

The per-draw values remain available for exact quantiles and joint summaries:

```@example streaming
quantile(compact.total, TreeDim(:ribbon, (0.1, 0.5, 0.9)); dims=:draw)
```

## Cache the tree itself with DynamicObjects

Load both packages to activate `TreeArraysDynamicObjectsExt`. A DO `@mmap`
property can return a `TreeData` directly, with or without a `::TreeData`
annotation. The generator and statistic bundle are ordinary indexed-property
arguments, so DO's callable/capture/configuration keys distinguish requests.
Use an explicit implementation version when the computation changes.

```julia
using TreeArrays, DynamicObjects

trajectory(draw, ntime) = [Float64(draw) + j / ntime for j in 1:ntime]
above(threshold) = y -> count(v -> v > threshold, y)

@dynamicstruct struct DrawStatistics
    cache_directory
    draw_labels
    ntime
    __cache_base__ = cache_directory

    @mmap v"1" compact(generator, statistics)::TreeData = map(TreeDim(:draw, draw_labels)) do draw
        y = generator(draw, ntime)
        TreeData(:stat => map(f -> fill(f(y)), statistics))
    end
end

job = DrawStatistics("my-cache", [1,2,3,4], 1024)
result = job.compact(trajectory, (;total=sum, above=above(2.5)))
quantile(result.total, TreeDim(:ribbon, (0.1, 0.5, 0.9)); dims=:draw)
```

On a cold-process hit, DO reopens the complete named result without invoking
the generator. Explicit full-curve requests can still return
`TreeData(trajectory(...), :time => labels)` from a named draw sweep; retaining
those arrays is a separate, explicit request.

## Supported storage and lifetime

The extension stores one tree container, with serialized structural metadata
and DO's existing DOMM codec for every numeric leaf. It preserves dim names,
coordinate collections, fixed positions, aggregated ghosts, record field names,
positional records, and ragged nesting (including empty children).

- Dense numeric arrays, numeric scalars, named/positional records, and arrays
  of `TreeData` are supported. Numeric element types are exactly those supported
  by DO's mmap codec. Unsupported terminals fail descriptively.
- Numeric array leaves reopen as readonly mapped `Array`s. Views and other
  numeric array wrappers are stored by value and reopen densely. Use
  `::TreeData` when the backing's concrete wrapper type changes; an incompatible
  concrete tree annotation is refused on load.
- Coordinates and other structural metadata are restored by Julia
  `Serialization`; they are ordinary in-memory metadata. Scalar terminals
  are restored by value. Numeric array leaves remain mapped after the file
  stream closes, for the lifetime of the returned arrays.
- DO validates the complete container before atomic publication. A failed
  write leaves the previous cache entry intact. Old mapped results keep their
  original backing when a new entry is published.

This is a Julia cache format governed by DO's implementation versions, not an
interchange format across arbitrary Julia/package versions. The public
`test/fixture_do_mmap_streaming.jl` exercises the optional DO integration in a
consumer environment, including separate-process reopen and exact numeric
oracles. The serial producer and quantile examples above execute in the docs.
