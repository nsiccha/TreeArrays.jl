# Named axes and records

Construct a tree, select coordinates by name, and transform its values while preserving their labels.

## Axes

Positional dimensions follow the backing array's axis order. A pair supplies coordinates; a bare symbol supplies only a name.

```@example axes
using TreeArrays, Tables
values = [1.0 2.0 3.0; 4.0 5.0 6.0]
X = TreeData(values, :draw => [101, 102], :time => [0.0, 0.5, 1.0])
(; shape=size(X), time=coords(X, :time), same_backing=parent(X) === values)
```

`parent(X)` is the original backing object. Use `coords(X, name)` for coordinates and `TreeArrays.dims(X)` for the dimensions declared on this node. An inner dimension of a ragged tree belongs to its child; it is not included in the parent's `dims` tuple.

Keyword arguments add fixed metadata rather than array axes. A reduced dimension also remains in the metadata, but no longer varies. Neither a fixed nor a reduced dimension has a coordinate vector for `coords` to return.

```@example axes
labelled = TreeData(values, :draw, :time => [0.0, 0.5, 1.0]; dose=10)
TreeArrays.dims(mean(labelled; dims=:draw))
```

## Selection

`selectdim` keeps the labels selected by a predicate, regular expression, boolean mask, or vector of positions. Its backing is a view. Regular expressions and predicates require coordinates; positional selectors also work on unlabelled axes.

```@example axes
early = selectdim(X, :time => t -> t <= 0.5)
(; time=coords(early, :time), values=Array(early))
```

Selection acts on an array axis on the current node. To select an inner axis of a ragged tree, select within its leaves. For a record, read the required field with property access.

## Records

A record stores fields that may themselves be arrays or trees. The pair form names its record axis. Fields already wrapped in `TreeData` retain their own dimensions; raw fields inherit the record's inner axes.

```@example axes
parameters = TreeData(:parameter => (initial=[10.0, 12.0], rate=[0.2, 0.3]),
                      :draw => [101, 102])
(; fields=propertynames(parameters), rates=parent(parameters.rate),
   record_axis=coords(outerdim(parameters)))
```

Use `TreeData` for construction. The aliases `TreeArray`, `TreeNamedTuple`, `TreeTuple`, and `TreeRaggedArray` identify backing layouts for dispatch; they are not separate constructors. A plain array of named tuples is not accepted as a record. Store fields as arrays, as above, or wrap each record in its own tree.

## Broadcasting

Broadcasting preserves dimensions and operates field by field on records. When two trees are combined, their varying axes must match by name, order, and coordinates. A dimension already reduced on one side can expand along the matching axis on the other.

```@example axes
centered = X .- mean(X; dims=:draw)
(; time=coords(centered, :time), values=Array(centered))
```

A plain array cannot supply dimension names for a tree-to-tree broadcast. Wrap it with the intended axes before combining it with a tree.

## Scenario grids

Mapping over a `TreeDim` creates a named sweep. Mapping over a product of dimensions passes the coordinate tuple to the function. A cell may return a scalar or a `TreeData`; wrap named-tuple results as records.

```@example axes
sweep = map(Iterators.product(TreeDim(:dose, [10, 20]),
                             TreeDim(:rate, [0.2, 0.4]))) do (dose, rate)
    dose / rate
end
Tables.rowtable(sweep)
```

The [concentration example](trajectories.md) uses named arrays inside a subject tree and keeps those labels through reductions and plotting.
