"""
    TreeDim(name::Symbol, values)
    TreeDim(name::Symbol)
    TreeDim(name => values)

A named dimension of a [`TreeData`](@ref). A tuple, vector, or range supplies
coordinates; the one-argument form creates an unlabelled array axis. Coordinates
are retained as provided. A scalar denotes fixed metadata; `nothing` denotes a
reduced dimension. Fixed and reduced dimensions add no physical array axis.
Use [`coords`](@ref) to read the coordinates of a labelled axis.
"""
struct TreeDim{N,M<:NamedTuple}
    meta::M
    TreeDim(N, meta::NamedTuple) = new{N,typeof(meta)}(meta)
end
TreeDim(N, values) = TreeDim(N, (;values))
TreeDim(N::Symbol) = TreeDim(N, missing)
TreeDim((N,values)::Pair) = TreeDim(N, values)
# `name`, `meta` and `sliced` are INTERNAL accessors (not exported, not part of the public
# API): read a dim's coordinates with `coords`, a tree's axes with `dims`, its backing
# with `parent`. Their names and the `meta` NamedTuple's layout may change without notice.
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

Wrap backing storage with named dimensions, without copying it. Positional
symbols or `name => coordinates` pairs describe physical axes in storage order.
Keyword arguments describe fixed metadata. The backing can be a numeric array,
a record, an array of trees, a tuple, or a scalar.

The pair form names a record axis; its fields become table columns and can be
read with property access. Raw fields inherit the record's inner dimensions;
fields already wrapped in `TreeData` retain their own dimensions. An array of
trees represents ragged nesting. Siblings must share layout and compatible
numeric element types, but their lengths may differ.

Use `parent(X)` to obtain the original backing. The four layout aliases below
are for dispatch; construct through `TreeData`. An array or tuple of plain
named tuples is rejected: use field arrays or wrap each record as a tree.
Calling `TreeData` with dimension names and no backing is an error.
"""
struct TreeData{P,M<:NamedTuple}
    parent::P
    meta::M
    TreeData(parent, meta::NamedTuple) = new{typeof(parent), typeof(meta)}(parent, meta)
end
Base.parent(X::TreeData) = getfield(X, :parent)
meta(X::TreeData) = getfield(X, :meta)
# `TreeData(:time)` -- dim names, no backing -- used to curry into a builder closure: a
# constructor call that silently returned a non-tree (todo iukt7; removed in 0.2.0, decision
# 1c2lsko). Throw by name and show the explicit spelling.
TreeData(dims::Symbol...) = error(
    "TreeArrays: `TreeData(" * join(repr.(dims), ", ") * ")` names dims but no backing array. " *
    "Pass the backing first (`TreeData(x, " * join(repr.(dims), ", ") * ")`); to build one tree " *
    "per item, map an explicit closure: `map((v, t) -> TreeData(v, :time => t), values, times)`."
)
TreeData(x, dims...; kwargs...) = TreeData(x, map(TreeDim, dims)..., map(TreeDim, keys(kwargs), values(kwargs))...)
TreeData(x, dims::TreeDim...) = TreeData(x, (;dims))
# An array of structs matches none of the four leaf shapes -- records are struct-of-arrays
# (`TreeNamedTuple`) -- and used to be accepted here only to fail late and opaquely inside a
# reduction (`float(::Type{Any})`). Refuse it by name at construction (decision 1c2lsko, 0.2.0);
# widening into a record-array leaf later would be non-breaking. Internal paths build through
# the inner `TreeData(parent, meta::NamedTuple)` constructor, which stays permissive.
# A `Tuple` of NamedTuples is the same shape in positional form (what a `map` sweep over a
# Tuple-valued dim builds when every cell is a plain NamedTuple), so it is refused alike.
const _StructsBacking = Union{AbstractArray{<:NamedTuple}, Tuple{NamedTuple, Vararg{NamedTuple}}}
TreeData(x::_StructsBacking, dims::TreeDim...) = error(
    "TreeArrays: a `$(typeof(x))` backing (an array or tuple of structs) is not a supported leaf: " *
    "records are stored struct-of-arrays. Build the record form " *
    "`TreeData(:field => (; a = …, b = …), dims...)`, or wrap each element in its own record " *
    "leaf: `TreeData(map(nt -> TreeData(:field => nt), x), dims...)`. To convert one value per draw " *
    "(a model's return values), give the array its dim names and let TreeArrays transpose it once: " *
    "`TreeData(returned(model, chain))` as is, or `TreeData(DimArray(x, (:draw, :chain)); axes = …)` " *
    "(DimensionalData extension)."
)
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

Dispatch alias for a [`TreeData`](@ref) backed by a named tuple. Construct with
`TreeData(name => fields, dims...)` to name the record axis. Read fields with
`X.field` and the record axis with [`outerdim`](@ref).
"""
TreeNamedTuple{P<:NamedTuple,M<:NamedTuple} = TreeData{P,M}

"""
    TreeRaggedArray

Dispatch alias for a [`TreeData`](@ref) backed by an array of trees. Array
operations describe the outer array; inner reductions operate within each
child. Outer reductions require matching kept axes or an explicit alignment
policy. See [`mapslices`](@ref).
"""
TreeRaggedArray{P<:AbstractArray{<:TreeData},M<:NamedTuple} = TreeData{P,M}

"""
    TreeArray

Dispatch alias for a [`TreeData`](@ref) backed by an `AbstractArray`, including
ragged trees at their outer level. Basic array operations forward to the backing:
`size`, `length`, `ndims`, `axes`, `eltype`, indexing, iteration, `collect`, and
`Array`. `TreeData` itself is not an `AbstractArray` subtype; use
[`TreeActualArray`](@ref) for a numeric array view.
"""
TreeArray{P<:AbstractArray,M<:NamedTuple} = TreeData{P,M}

"""
    TreeTuple

Dispatch alias for a [`TreeData`](@ref) backed by a tuple, such as the output
of a quantile reduction with tuple-valued levels.
"""
TreeTuple{P<:Tuple,M<:NamedTuple} = TreeData{P,M}
# convenience accessors (avoid spelling out `meta(...).field` everywhere)
# `dims` collides with the `dims=` kwarg used throughout mapslices/quantile, so inside
# those method bodies it is self-qualified as `TreeArrays.dims(...)` (decision 2jzn6o).
"""
    dims(X::TreeData)
    dims(A::TreeActualArray)

Return the dimensions declared on this node in declaration order, including
fixed and reduced dimensions. This does not recursively collect dimensions
from children. Use [`coords`](@ref) for a labelled axis's coordinates. Inside
functions with a `dims` keyword, qualify the call as `TreeArrays.dims(X)`.
"""
dims(X::TreeData) = meta(X).dims                 # the tree's (inner) axes

"""
    outerdim(X::TreeNamedTuple)

Return the record axis whose coordinates are the field names. The record must
have been built with the named pair form of [`TreeData`](@ref); a raw named-tuple
backing without a named record axis does not supply one.
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
# A TreeTuple is EITHER a positional-record LEAF (a tuple of scalars -- a quantile band result)
# OR a positional CONTAINER whose elements are themselves TreeData (what `map(f, ::TreeDim)`
# builds -- the single-axis twin of a product-swept TreeRaggedArray). `eltype(parent(X))` answers
# the leaf case but returns the CHILD TREE type for the container case, so a reducer's
# `float(_eltype(X))` / `_eltype(X)[]` then chokes on a `TreeArray` type (snag nanquantile-tree).
# Recurse through a representative child instead -- exactly as TreeNamedTuple/TreeRaggedArray do;
# for a scalar-tuple leaf `_eltype(first(parent(X)))` is the first field's type, matching the
# established "first field" convention above. The empty tuple has no representative, so the TYPE
# walk answers (Union{} for a zero-field tuple, preserving the instance contract this replaces).
_eltype(X::TreeTuple)      = isempty(parent(X)) ? _eltype(typeof(parent(X))) : _eltype(first(parent(X)))
_eltype(X::TreeNamedTuple) = _eltype(first(parent(X)))
_eltype(X::TreeRaggedArray) = isempty(parent(X)) ?
    _eltype(eltype(parent(X))) :      # no representative -- ask the element type
    _eltype(first(parent(X)))         # a representative exists, and it always answers
# A SCALAR-parent leaf -- what `quantile(X, TreeDim(:median, 0.5))` builds at each position --
# is none of the four aliases, and used to fall to the `eltype(x)` fallback below: `Any` for
# every TreeData, which a chained `nanquantile`'s `float(_eltype(X))` cannot take. Its leaf
# eltype is the scalar's own TYPE, the terminal `_actualeltfield` and the type walk's `where P`
# method already use (a `String` leaf is `String`, not `eltype(String) == Char`). A TreeData
# parent is a wrapper level, so that one recurses by instance like the aliases above.
_eltype(X::TreeData) = _leafeltype(parent(X))
_leafeltype(x::TreeData) = _eltype(x)
_leafeltype(x) = typeof(x)
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
