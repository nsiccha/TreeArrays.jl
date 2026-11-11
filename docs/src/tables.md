# Interoperability

Choose a table or numeric array view for a downstream consumer, and convert DimensionalData or FlexiChains inputs into trees.

## Tables

A tree implements Tables.jl. Array axes become long coordinate columns; record fields become wide value columns. `TreeTable` additionally pivots an explicitly chosen axis.

```@example interoperability
using TreeArrays, Tables
X = TreeData([1.0 2.0; 3.0 4.0], :draw => [101, 102], :parameter => [:a, :b])
(; long=Tables.rowtable(X), wide=Tables.rowtable(TreeTable(X; wide=:parameter)))
```

The table view does not choose an analysis: it changes the layout presented to a consumer. `Tables.rows` and `Tables.columns` provide lazy access; `Tables.rowtable` materializes rows, as used here to display the result. A ragged axis cannot be pivoted wide. Keep it long or choose an alignment and reduction first.

A record already has named columns. For array-valued fields, wrap each field with its own axes. Container metadata alone does not supply row axes to the table adapter.

```@example interoperability
draw_axis = TreeDim(:draw, [101, 102])
fields = map(v -> TreeData(v, draw_axis), (a=[1.0, 3.0], b=[2.0, 4.0]))
record = TreeData(:parameter => fields, draw_axis)
Tables.schema(TreeTable(record))
```

The [quantile layouts](reductions.md#Quantile-layouts) produce either a record or an explicit output axis. The [concentration figure](trajectories.md#Figure) consumes the record's table directly.

## Numeric arrays

Use `TreeActualArray` when a consumer requires an `AbstractArray`. It presents homogeneous records as numeric arrays, with the record axis following the inner axes. It also supports dense leaves and grids of homogeneous numeric scalar leaves. The view is read-only; `Array(view)` makes a mutable copy.

```@example interoperability
view = TreeActualArray(record)
(; shape=size(view), values=Array(view), same_tree=parent(view) === record)
```

Changing a backing value is visible through the view. Keep the tree's shape, coordinates, and element types fixed while a view exists.

```@example interoperability
before = view[1, 1]
parent(record.a)[1] = 9.0
(; before, after=view[1, 1])
```

For array-backed ragged leaves, the constructor requires an alignment dimension and policy. `:truncate_min` selects each leaf's first positions up to the shortest length; `:thin_min` selects evenly spread positions; `:error_equal` requires equal lengths. Labels at the selected positions must agree across leaves. These policies select observations rather than interpolating them.

```@example interoperability
chains = TreeData([TreeData([1.0, 2.0, 3.0], :draw),
                   TreeData([4.0, 5.0], :draw)], :chain => [1, 2])
aligned = TreeActualArray(chains; dims=:draw, align=:truncate_min)
(; shape=size(aligned), values=Array(aligned))
```

The outer sibling axis follows the aligned inner axis. For chain matrices with a parameter axis this gives the draw-by-chain-by-parameter order used by numeric sampling diagnostics. Selecting a subset changes the samples used by those diagnostics; choose that subset deliberately.

## DimensionalData

Loading DimensionalData enables its constructor extension. Numeric cells retain their backing, coordinate labels are carried over, and `:iter` becomes `:draw`.

```@example interoperability
using DimensionalData: DimArray, Dim
labelled = DimArray(reshape(Float64.(1:6), 3, 2),
                   (Dim{:iter}(101:103), Dim{:parameter}([:a, :b])))
converted = TreeData(labelled)
(; draws=coords(converted, :draw), same_backing=parent(converted) === parent(labelled),
   values=Array(converted))
```

Array or named-tuple cells are converted into nested tree storage. Supply the constructor's `axes` keyword to name inner axes; its structure follows the cell's fields. Numeric-cell conversion and structured-cell conversion therefore have different storage behavior.

## FlexiChains

Loading FlexiChains enables conversion in both directions. Parameters become a named record; extra sampler fields are excluded unless `extras=true`. Conversion reads the chain's parameter arrays; it is not a promise that the original chain storage is shared.

```@example interoperability
using FlexiChains: FlexiChain, Parameter, Extra
chain = FlexiChain{Symbol}(reshape(Float64.(1:24), 4, 2, 3),
                          (Parameter(:a), Parameter(:b), Extra(:lp)))
posterior = TreeData(chain)
restored = FlexiChain(posterior)
(; fields=propertynames(posterior),
   array_shape=size(TreeActualArray(posterior)),
   roundtrip=Array(TreeActualArray(TreeData(restored))) == Array(TreeActualArray(posterior)))
```

The reverse conversion requires rectangular numeric data with compatible draw and chain axes. Ragged or mixed nonnumeric fields must be resolved before that boundary.

The [concentration example](trajectories.md) uses the table interface to pass a ragged tree's summaries to AlgebraOfVega.
