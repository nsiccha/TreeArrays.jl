```@meta
CurrentModule = TreeArrays
```

# API

## Public API (frozen for 1.0)

Everything in this section is public: documented, tested, and covered by
semantic versioning. Before 1.0 the minor digit is the breaking one; from 1.0 a
breaking change to anything listed here needs a major bump, and the
[changelog](https://github.com/nsiccha/TreeArrays.jl/blob/main/CHANGELOG.md)
names every change. Anything **not** listed here is internal and may change
without notice: every `_`-prefixed name, `TreeArrays.meta`, `TreeArrays.name`,
`TreeArrays.sliced`, and the layout of a tree's or a dim's `meta` NamedTuple.
A test (`test/public_api.jl`) holds the export list to exactly this surface.

### Types and construction

- [`TreeDim`](@ref): `TreeDim(name, values)`, `TreeDim(name)`,
  `TreeDim(name => values)`. Coordinates are kept exactly as provided.
- [`TreeData`](@ref): `TreeData(backing, dims...; kwarg_dims...)`,
  `TreeData(name => NamedTuple, dims...)` (a record with a named record axis),
  `TreeData(X::TreeData, dims...)` (append dims).
- `TreeData(A::AbstractDimArray; axes = nothing, kwarg_dims...)` (DimensionalData
  extension): A's dims become named axes with their lookups as labels (`:iter`
  becomes `:draw`). Number cells stay a zero-copy leaf. One NamedTuple or array per
  cell (a model's return values, `returned(model, chain)`) is converted once into
  records, dense leaves and ragged nodes; `axes` names the inner axes, mirroring
  the value's structure (`(pe = (:participant, :trial),)`).
- The four dispatch aliases [`TreeArray`](@ref), [`TreeNamedTuple`](@ref),
  [`TreeTuple`](@ref), [`TreeRaggedArray`](@ref). They are for dispatch only;
  always construct through `TreeData`. Dispatching on `TreeData{P}`'s first
  parameter (the backing type) is supported; the second parameter is internal.
- [`TreeActualArray`](@ref): `TreeActualArray(X)` and
  `TreeActualArray(X::TreeRaggedArray; dims, align)`.
- [`TreeTable`](@ref): `TreeTable(X; wide = ())`.

### Reading a tree

- [`dims`](@ref), [`outerdim`](@ref), [`coords`](@ref) (`coords(d)` and
  `coords(X, name)`), and `parent(X)` for the zero-copy backing.
- `X.field` on a record (`Base.getproperty`) and `propertynames(X)`; on a ragged
  tree of records `.field` maps over the leaves.
- On a dense leaf the basic array interface is forwarded to the backing:
  `size`, `length`, `ndims`, `axes`, `eltype`, `getindex`, `iterate`.
  `collect(X)` and `Array(X)` copy the backing out. At the outer level of a
  ragged tree the same methods answer about the outer axis (count, sub-trees).
- `size(X, name)` and `axes(X, name)` on every tree shape: by dim name. A name
  inside a ragged tree answers when every child agrees; reduced and fixed dims
  have no length.
- Broadcasting (`f.(X)`, `X .- Y`) returns a tree with the same dims. Trees combine
  when their axes match by name, order and labels; a reduced or fixed dim
  stretches along the other tree's axis; records go field by field, ragged trees
  leaf by leaf. A plain array refuses by name.
- `show` for text, `text/html` and `text/markdown`, all bounded previews.

### Reductions and selection

- [`mapslices`](@ref): `mapslices(f, X; dims, coords = false, threads = false, align = nothing)`.
- [`mean`](@ref), [`sum`](@ref), [`std`](@ref), [`var`](@ref):
  `(X; dims = nothing, threads = true, align = nothing)`; `std` and `var` take
  `corrected = true`.
- [`quantile`](@ref): `quantile(X, TreeDim(name, levels); dims)`,
  `quantile(X, p::NamedTuple; dims)`, `quantile(X, name => p::NamedTuple; dims)`,
  each with `threads = true, align = nothing`. `nanquantile` has the same three forms through the
  NaNStatistics extension.
- [`hdi`](@ref): `hdi(y, mass = 0.9)` and `hdi(X; dims, mass = 0.9, threads = true, align = nothing)`.
- [`@kernel`](@ref): `@kernel (:reduces => :into) function f(slice) ... end`,
  the `(slice, coords)` form, and the post-hoc form on an existing function.
- [`selectdim`](@ref): `selectdim(X, name => selector)` and
  `selectdim(X, name, selector)`.
- Scenario sweeps: `map(f, d::TreeDim)` and
  `map(f, Iterators.product(dims::TreeDim...))`, with `stack` accepted as a
  spelling of the same thing.

### Tables

- `Tables.columns`, `Tables.rows` and `Tables.schema` on a `TreeData` (the
  long melt; ragged trees melt long) and on a `TreeTable` (the `wide =` pivot,
  which refuses a ragged axis by name).

### Extensions

- FlexiChains: `TreeData(chain; extras = false)`, `FlexiChain(X)`,
  `TreeData(summary::FlexiSummary)`.
- DynamicObjects: `save(Val(:mmap), path, X)` and `load(Val(:mmap), path, T)`
  for mmap-backed caches.
- NaNStatistics: `nanquantile`, as above.

### Settled before 1.0 (0.2.0)

Three corners were decided and closed in 0.2.0:

- `setdim` is gone. It was an exported stub that errored on every call; a
  lazy re-label / fix-a-dim operation can be added later without a breaking
  change.
- `TreeData(:time)` with no backing throws by name. It used to return a
  builder closure for `map`; write `map((v, t) -> TreeData(v, :time => t), values, times)`.
- An `AbstractArray{<:NamedTuple}` backing (an array of structs), or a `Tuple`
  of `NamedTuple`s, is refused by name at construction. Records are stored
  struct-of-arrays: use
  `TreeData(:field => (; a = …, b = …), dims...)`, or wrap each element in its
  own record leaf. The same refusal fires inside `mapslices` and the sweeps
  when a kernel or cell returns a plain `NamedTuple`: return
  `TreeData(:stat => nt)` instead.

### Settled design points

- No eager restructuring: a reduction never pivots, stacks or `vcat`s, and
  `TreeData` stores its `parent` verbatim.
- Never a valid-looking wrong value: `dims =`, `coords(X, name)` and `.field`
  on ragged records resolve every name or throw by name.
- Built-in reducers thread by default; `mapslices` and `@kernel` opt in with
  `threads = true`.
- `TreeTable(wide =)` refuses a ragged axis by name; melt a ragged tree long.

## Reference

Everything `TreeArrays` exports, plus the `Base` and `Statistics` methods it
extends.

```@index
```

```@autodocs
Modules = [TreeArrays]
Order = [:module, :type, :macro, :function]
```
