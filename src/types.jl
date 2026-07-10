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
TreeNamedTuple{P<:NamedTuple,M<:NamedTuple} = TreeData{P,M}
TreeRaggedArray{P<:AbstractArray{<:TreeData},M<:NamedTuple} = TreeData{P,M}
TreeArray{P<:AbstractArray,M<:NamedTuple} = TreeData{P,M}
TreeTuple{P<:Tuple,M<:NamedTuple} = TreeData{P,M}
# convenience accessors (avoid spelling out `meta(...).field` everywhere)
# `dims` collides with the `dims=` kwarg used throughout mapslices/quantile, so inside
# those method bodies it is self-qualified as `TreeArrays.dims(...)` (decision 2jzn6o).
dims(X::TreeData) = meta(X).dims                 # the tree's (inner) axes
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
