"""
    TreeDim(name::Symbol, values)
    TreeDim(name::Symbol)
    TreeDim(name => values)

One named axis of a [`TreeData`](@ref). The axis *name* lives in the type, so
reductions that name it (`dims = :time`) resolve statically; the *coordinates*
live in the instance and are kept **exactly as provided**.

`values` decides what kind of dim this is:

| `values` | kind | meaning |
|---|---|---|
| a `Tuple` / `AbstractVector` / `AbstractRange` | **labelled axis** | a real axis; its elements are the coordinates |
| `missing` (the 1-arg form) | **unlabelled axis** | a real axis with no coordinates |
| any scalar (`dose = 20`) | **fixed position** | one coordinate, contributing *no* array axis |
| `nothing` | **aggregated ghost** | what a reduction leaves behind in place of the axis it collapsed |

A `Tuple` stays a `Tuple`, a `Vector` stays a `Vector`, a range stays a range —
the same keep-as-provided convention `quantile`'s levels and [`selectdim`](@ref)'s
subsets follow. Read the coordinates back with [`coords`](@ref); never with
`collect`, which answers `Any[missing]` / `Any[nothing]` for the two kinds that
have no coordinates at all.

# Examples
```julia
TreeDim(:time, range(0, 1, 100))        # labelled axis
TreeDim(:draw)                          # unlabelled axis
TreeDim(:ribbon, (0.025, 0.5, 0.975))   # the output axis a `quantile` reduction writes into
TreeDim(:dose, 20)                      # a fixed position, not an axis
```

See also [`TreeData`](@ref), [`coords`](@ref), [`dims`](@ref).
"""
struct TreeDim{N,M<:NamedTuple}
    meta::M
    TreeDim(N, meta::NamedTuple) = new{N,typeof(meta)}(meta)
end
TreeDim(N, values) = TreeDim(N, (;values))
TreeDim(N::Symbol) = TreeDim(N, missing)
TreeDim((N,values)::Pair) = TreeDim(N, values)
name(::TreeDim{N}) where N = N
name(::Type{<:TreeDim{N}}) where N = N
meta(X::TreeDim) = getfield(X, :meta)
sliced(X::TreeDim) = TreeDim(name(X), (;values=nothing))
# keep-as-provided: a collection value is a real axis (iterate its coords); a scalar,
# `missing` (unlabelled) or `nothing` (aggregated) value is a single fixed position.
_coords(v::Union{Tuple,AbstractArray,AbstractRange}) = v
_coords(v) = (v,)
Base.length(X::TreeDim) = length(_coords(meta(X).values))
Base.iterate(X::TreeDim, args...) = iterate(_coords(meta(X).values), args...)
"""
    TreeData(backing, dims...; kwarg_dims...)
    TreeData(name => namedtuple, dims...)
    TreeData(dims::Symbol...)

A named, hierarchical, possibly-ragged array over flat backing storage.

A `TreeData` pairs a `parent` — the backing storage, never copied — with a tuple
of named axes ([`TreeDim`](@ref)). Positional arguments become dims, in the order
of the backing array's axes; a `name => values` pair gives that axis its
coordinates, and keyword arguments become fixed dims.

```julia
X = TreeData(randn(1000, 179, 100), :draw, :subject, :time => range(0, 1, 100))
D = TreeData(randn(1000, 8), :draw, :param; placebo = :on)   # `placebo` is a fixed dim
```

Four leaf shapes are distinguished by the *type* of `parent`. They are
dispatch-only aliases — you always construct through `TreeData`:

| Alias | `parent` is… | meaning |
|---|---|---|
| [`TreeArray`](@ref) | an `AbstractArray` of numbers | a dense leaf |
| [`TreeNamedTuple`](@ref) | a `NamedTuple` | a record; carries a separate record axis, see [`outerdim`](@ref) |
| [`TreeTuple`](@ref) | a `Tuple` | a positional record |
| [`TreeRaggedArray`](@ref) | an `AbstractArray{<:TreeData}` | a ragged nesting of sub-trees of differing shape |

Build a record with the `name => NamedTuple` form, which names the field axis:

```julia
TreeData(:param => (; alpha = a_matrix, beta = b_matrix), :draw, :chain)
```

!!! warning "Currying"
    `TreeData(:time)` — dim names and **no** backing array — returns a *builder*
    closure `(x, vals...) -> TreeData(x, :time => vals)`, meant for `map`:

    ```julia
    measurement = map(TreeData(:time), per_subject_values, per_subject_times)
    ```

    It is not a constructed tree; do not pass it where you meant one.

A `TreeData` is deliberately **not** an `AbstractArray` (only the array-backed
[`TreeArray`](@ref) leaf forwards `size`/`getindex`/`iterate` to its parent). To
feed a numeric-array API, wrap it in [`TreeActualArray`](@ref); to feed a
Tables.jl consumer, use [`TreeTable`](@ref).

See also [`mapslices`](@ref), [`quantile`](@ref), [`selectdim`](@ref),
[`dims`](@ref), [`coords`](@ref).
"""
struct TreeData{P,M<:NamedTuple}
    parent::P
    meta::M
    TreeData(parent, meta::NamedTuple) = new{typeof(parent), typeof(meta)}(parent, meta)
end
Base.parent(X::TreeData) = getfield(X, :parent)
meta(X::TreeData) = getfield(X, :meta)
TreeData(dims::Symbol...) = (x, vals...)->TreeData(x, map(TreeDim, dims, vals)...)
TreeData(x, dims...; kwargs...) = TreeData(x, map(TreeDim, dims)..., map(TreeDim, keys(kwargs), values(kwargs))...)
TreeData(x, dims::TreeDim...) = TreeData(x, (;dims))
# a TreeNamedTuple keeps its record axis in the type as a SEPARATE `outer_dim` (which axis
# enumerates the fields is a property of the container, not of a dim) *and* lists it in
# `dims` like any other axis, so it's not lost when forwarded (e.g. by `TreeData(X::TreeData, dims...)`).
TreeData((name, X)::Pair{Symbol,<:NamedTuple}, dims::TreeDim...) = begin
    rec = TreeDim(name, keys(X))
    TreeData(X, (;dims = (dims..., rec), outer_dim = rec))
end
TreeData(X::TreeData, dims::TreeDim...) = TreeData(parent(X), merge(meta(X), (;dims=(meta(X).dims..., dims...))))
"""
    TreeNamedTuple

Dispatch alias for a [`TreeData`](@ref) whose `parent` is a `NamedTuple` — a
**record**. Its fields are its record axis; read them with `X.fieldname`
(zero-copy: a field that is already a `TreeData` comes back as is, a raw field is
wrapped with the container's inner axes) and name that axis with
[`outerdim`](@ref).

Construct one with the pair form so the record axis is named:
`TreeData(:param => (; alpha = …, beta = …), :draw, :chain)`.

A record's fields melt to **wide** columns (one column per field) — see
[`TreeTable`](@ref).
"""
TreeNamedTuple{P<:NamedTuple,M<:NamedTuple} = TreeData{P,M}

"""
    TreeRaggedArray

Dispatch alias for a [`TreeData`](@ref) whose `parent` is an array of
[`TreeData`](@ref) — a **ragged** nesting: one outer axis over sub-trees whose
shapes need not agree.

```julia
subjects = TreeData(map(s -> TreeData(view(mat, :, idx[s]), :draw, :time), 1:n), :subject)
```

`mapslices(f, X; dims = :time)` reduces each sub-tree over *its own* `:time`
axis; reducing the outer axis pushes it down to the leaves and recurses. Nothing
is stacked or materialized at any point. A ragged tree also melts to a **long**
table (row count `sum(length, leaves)`), so it need not be made rectangular
before plotting — but [`TreeTable`](@ref)'s `wide=` pivot does require
rectangularity and refuses a ragged axis by name.
"""
TreeRaggedArray{P<:AbstractArray{<:TreeData},M<:NamedTuple} = TreeData{P,M}

"""
    TreeArray

Dispatch alias for a [`TreeData`](@ref) whose `parent` is an `AbstractArray` — a
**dense leaf**. This is the only shape that forwards the basic array interface
(`size`, `length`, `ndims`, `axes`, `getindex`, `iterate`, `collect`, `Array`) to
its backing array; a `TreeData` is still not an `AbstractArray` subtype.

Because `TreeRaggedArray{P<:AbstractArray{<:TreeData}}` is a strict
subconstraint, those methods also apply to a ragged value at the **outer** level:
`size` is the outer count and `getindex` returns a sub-tree.
"""
TreeArray{P<:AbstractArray,M<:NamedTuple} = TreeData{P,M}

"""
    TreeTuple

Dispatch alias for a [`TreeData`](@ref) whose `parent` is a `Tuple` — a
**positional record**. This is what a `quantile` reduction over a `Tuple` of
levels leaves on the leaf.
"""
TreeTuple{P<:Tuple,M<:NamedTuple} = TreeData{P,M}
# convenience accessors (avoid spelling out `meta(...).field` everywhere)
# `dims` collides with the `dims=` kwarg used throughout mapslices/quantile, so inside
# those method bodies it is self-qualified as `TreeArrays.dims(...)` (decision 2jzn6o).
"""
    dims(X::TreeData)
    dims(A::TreeActualArray)

The tree's axes, as a tuple of [`TreeDim`](@ref)s — including fixed dims and the
aggregated ghosts reductions leave behind, in declaration order.

!!! note "Self-qualify inside method bodies"
    The bare name `dims` collides with the `dims =` keyword argument used
    throughout [`mapslices`](@ref)/[`quantile`](@ref), so inside a function that
    takes one, write `TreeArrays.dims(X)`.

To read one axis's coordinates, prefer [`coords`](@ref).
"""
dims(X::TreeData) = meta(X).dims                 # the tree's (inner) axes

"""
    outerdim(X::TreeNamedTuple)

The record axis of a [`TreeNamedTuple`](@ref) — the [`TreeDim`](@ref) whose
coordinates are the record's field names.

Only a tree built through the pair form (`TreeData(:param => (; a = …, b = …),
…)`) carries one; handing a `NamedTuple` straight to `TreeData(x, dims...)` does
not name a record axis, and this errors.
"""
outerdim(X::TreeData) = meta(X).outer_dim        # a TreeNamedTuple's record axis

# leaf numeric eltype: recurse through NamedTuple / ragged nesting down to the backing array.
# TreeNamedTuple uses the FIRST field's type -- fine for quantile's homogeneous numeric records.
#
# ELEMENT when there is one, TYPE when there is not, named error when there is neither.
#
# The instance walk (`_eltype(first(parent(X)))`) is undefined on a zero-length ragged nesting
# -- the very thing a product-mapped sweep emits for a cell with no data -- and threw a bare
# `BoundsError` from inside `quantile`/`nanquantile`'s `scratch = _eltype(X)[]`. At an EMPTY
# boundary the element TYPE answers instead, and it can, provided the container carries a
# concrete one (`TreeData(typeof(leafproto)[], …)`).
#
# But the type walk must NOT take over the non-empty case, which is why only the empty branch
# below reaches it. A JAGGED tree -- siblings whose axis lengths differ AS TYPES -- has a
# non-concrete element type that carries no leaf structure at all, while any one of its
# elements carries it fine. Reducing exactly such a tree is what TreeArrays is FOR (ragged
# per-subject arrays without stacking), and a type-only walk turns it into an error. Nesting
# makes that worse: an outer container's type walk cannot reach an inner instance to recover.
_eltype(X::TreeArray)      = eltype(parent(X))
_eltype(X::TreeTuple)      = eltype(parent(X))
_eltype(X::TreeNamedTuple) = _eltype(first(parent(X)))
_eltype(X::TreeRaggedArray) = isempty(parent(X)) ?
    _eltype(eltype(parent(X))) :      # no representative -- ask the element type
    _eltype(first(parent(X)))         # a representative exists, and it always answers
_eltype(x) = eltype(x)

# --- the TYPE walk. Reached only from the empty branch above, where no element exists.
function _eltype(::Type{T}) where T<:TreeData
    # `TreeData(TreeData[], ...)` -- the natural-looking way to spell an empty container --
    # erases the child structure this walk reads: `fieldtype(TreeData, :parent)` widens to
    # `Any`, and the leaf eltype then silently degrades to `Any`, surfacing much later as an
    # opaque `MethodError: float(::Type{Any})` from inside a reducer. Fail here, with the
    # spelling that works. Mirrors `_schema`'s identical guard (tables.jl), so the reduce
    # path and the melt path reject the same value for the same stated reason.
    isconcretetype(T) || error("TreeArrays: `$T` is not a concrete TreeData type -- an empty container must carry its element type (`TreeData(typeof(leafproto)[], :assay_name => String[])`, not `TreeData(TreeData[], ...)`), because the leaf eltype is derived from the element TYPE alone")
    _eltype(fieldtype(T, :parent))
end
_eltype(::Type{P}) where P<:AbstractArray{<:TreeData} = _eltype(eltype(P))   # ragged nesting
_eltype(::Type{P}) where P<:AbstractArray            = eltype(P)             # dense leaf
# `Tuple{TreeData,Vararg{TreeData}}`, not `Tuple{Vararg{TreeData}}`: the latter also matches
# the EMPTY `Tuple{}`, whose `eltype` is `Union{}` -- not a TreeData to recurse into. Spelling
# "at least one" lets a zero-field TreeTuple fall through to the plain-tuple method below and
# keep returning `Union{}`, exactly as the instance walk this replaces did.
_eltype(::Type{P}) where P<:Tuple{TreeData,Vararg{TreeData}} = _eltype(eltype(P))
_eltype(::Type{P}) where P<:Tuple                    = eltype(P)
_eltype(::Type{P}) where P<:NamedTuple               = _eltype(fieldtype(P, 1))
_eltype(::Type{P}) where P                           = P                     # scalar terminal
