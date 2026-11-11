```@meta
CurrentModule = TreeArrays
```

# API

## Construction

```@autodocs
Modules = [TreeArrays]
Pages = ["types.jl"]
Filter = x -> x in (TreeDim, TreeData, TreeArray, TreeNamedTuple, TreeTuple, TreeRaggedArray)
```

## Dimensions

```@docs
dims
outerdim
coords
```

## Reductions

```@autodocs
Modules = [TreeArrays]
Pages = ["reducers.jl", "mapslices.jl", "kernel.jl"]
```

## Selection and sweeps

```@autodocs
Modules = [TreeArrays]
Pages = ["selectdim.jl", "sweeps.jl", "properties.jl"]
```

## Views

```@docs
TreeTable
TreeActualArray
```

## Extensions

Load the companion package to activate its methods. Executed recipes live in the linked guides.

| Package | Entry points | Guide |
| --- | --- | --- |
| DimensionalData | `TreeData(A; axes=nothing)` | [Interoperability](tables.md#DimensionalData) |
| FlexiChains | `TreeData(chain; extras=false)`, `FlexiChain(X)`, `TreeData(summary)` | [Interoperability](tables.md#FlexiChains) |
| NaNStatistics | `nanquantile(X, probabilities; dims)` | [Reductions](reductions.md#Missing-numeric-values) |
| DynamicObjects | `save(Val(:mmap), path, X)`, `load(Val(:mmap), path, TreeData)` | [Streaming and caches](streaming.md#Mapped-storage) |
