# ===================== Tables.jl integration =====================
# A reduced TreeData is a lazy Tables.jl COLUMN source: nothing melts at
# mapslices/quantile construction time (eager compute / lazy assembly,
# decision 1uzarfr) -- the tree -> columns melt happens exactly once, at the
# `Tables.columns` export boundary (decision 14snrv9 + the AoV contract,
# 2026-07-06: AoV needs a concrete, indexable `AbstractVector` per pulled
# column -- no streaming). `Tables.columnnames`/`Tables.schema` are computed
# from the TYPE alone (no melt, cheap).
#
# Supported shape (errors clearly otherwise -- never a silently-wrong table):
# recurses over the SAME structural cases `_leafreduce` already dispatches on
# (mapslices.jl): TreeRaggedArray (outer array of TreeData -- recurse per
# element), TreeArray / TreeTuple (array/tuple-backed leaf -- one row per
# position, `:value` column), TreeNamedTuple (record -- outer_dim's name
# becomes a column holding the field key; every field must melt to an
# IDENTICAL column schema), and a generic `TreeData` fallback (a bookkeeping
# wrapper around another TreeData, or a bare scalar leaf).
#
# Ghost/missing/fixed dim -> column mapping (decision oni1bc):
#   `sliced` ghost (values === nothing)   -> dropped, no coordinate to record
#   unlabelled axis (values === missing)  -> column kept, value = 1-based position
#   fixed scalar (e.g. quantile's scalar p, or a fixed `random_effect=:on` dim)
#                                          -> column kept, constant every row
#   (a stray real axis with no enclosing array/tuple/record position is
#   unsupported -- there's nowhere to read a per-row coordinate from)

using Tables
using FillArrays: Fill

const MELT_COUNT = Ref(0)   # test-only instrumentation: counts Tables.columns melt invocations

# ---- dim classification: axis (real per-position coordinate) / ghost
#      (dropped) / fixed (constant column) -- a 3-way refinement of `_isaxis`.
#      Works identically on a TreeDim INSTANCE or its TYPE (both dispatch
#      through `_isaxis`, which already supports both).
_dimkind(d::TreeDim) = _isaxis(d) ? :axis : (meta(d).values === nothing ? :ghost : :fixed)
_dimkind(::Type{T}) where {N,M,T<:TreeDim{N,M}} = _isaxis(T) ? :axis : (fieldtype(M, :values) === Nothing ? :ghost : :fixed)

_dimvalue(d::TreeDim, i::Int) = _dimvalue(meta(d).values, i)
_dimvalue(::Missing, i::Int) = i
_dimvalue(v::Union{Tuple,AbstractArray,AbstractRange}, i::Int) = v[i]

_fixedtype(::Type{T}) where {N,M,T<:TreeDim{N,M}} = fieldtype(M, :values)

_axistype(::Type{T}) where {N,M,T<:TreeDim{N,M}} = _axisvaltype(fieldtype(M, :values))
_axisvaltype(::Type{Missing}) = Int
_axisvaltype(V::Type{<:Union{Tuple,AbstractArray,AbstractRange}}) = eltype(V)

# positions along a node's own leading real axes -- AbstractArray uses
# CartesianIndices (mirrors `_reduceouter`'s array handling); Tuple (a
# TreeTuple leaf) is a plain 1:n linear sequence.
_positions(p::AbstractArray) = CartesianIndices(p)
_positions(p::Tuple) = 1:length(p)
_nax(::Type{<:AbstractArray{<:Any,N}}) where N = N
_nax(::Type{<:Tuple}) = 1

# a dims tuple (of TreeDim instances OR types -- `_dimkind` works on both)
# with NO positional axis of its own: every entry must be ghost (dropped) or
# fixed (kept); a real axis here has no per-row position to read from.
function _flatfixed(alldims)
    any(d -> _dimkind(d) === :axis, alldims) &&
        error("TreeArrays Tables adapter: a real axis dim among $(map(name, alldims)) has no enclosing array/tuple/record position -- unsupported shape")
    filter(d -> _dimkind(d) === :fixed, alldims)
end

# split a node's own `dims(X)` tuple into (leading real per-position axes,
# trailing fixed dims [ghosts dropped]) at the parent-array-axis boundary --
# mirrors `_reduceouter`'s positional-axis invariant.
function _splitmelt(alldims::Tuple, nax::Int)
    axdims, rest = alldims[1:nax], alldims[nax+1:end]
    all(d -> _dimkind(d) === :axis, axdims) ||
        error("TreeArrays Tables adapter: expected the first $nax dims to be real array axes (positionally), got $(map(name, axdims)) with kinds $(map(_dimkind, axdims))")
    axdims, _flatfixed(rest)
end

# ---- schema: (names, types) from the TYPE alone, no melt. ----
function _schema(::Type{T}) where T<:TreeData
    P = fieldtype(T, :parent)
    alldims = Tuple(fieldtype(fieldtype(T, :meta), :dims).parameters)
    names, types = _ownschema(T, P, alldims)
    cnames, ctypes = _childschema(T, P)
    (Tuple(vcat(names, collect(cnames))), Tuple(vcat(types, collect(ctypes))))
end

function _ownschema(::Type{T}, ::Type{P}, alldims) where {T<:TreeData,P<:Union{AbstractArray,Tuple}}
    axdims, fixed = _splitmelt(alldims, _nax(P))
    (Symbol[map(name, axdims)..., map(name, fixed)...],
     Type[map(_axistype, axdims)..., map(_fixedtype, fixed)...])
end
function _ownschema(::Type{T}, ::Type{P}, alldims) where {T<:TreeData,P<:NamedTuple}
    recname = name(fieldtype(fieldtype(T, :meta), :outer_dim))
    fixed = _flatfixed(filter(d -> name(d) !== recname, alldims))
    (Symbol[map(name, fixed)...], Type[map(_fixedtype, fixed)...])
end
function _ownschema(::Type{T}, ::Type{P}, alldims) where {T<:TreeData,P}   # bookkeeping wrapper (P<:TreeData) or scalar leaf
    fixed = _flatfixed(alldims)
    (Symbol[map(name, fixed)...], Type[map(_fixedtype, fixed)...])
end

_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:AbstractArray{<:TreeData}} = _schema(eltype(P))
_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:AbstractArray} = ((:value,), (eltype(P),))
# a Tuple-backed axis whose ELEMENTS are themselves TreeData (e.g. a chained
# quantile whose levels container was a Tuple, not a Vector -- decision
# xxmv6c) is the Tuple-backed analogue of TreeRaggedArray: recurse per
# position, same as the array-of-TreeData case above.
_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:Tuple{Vararg{<:TreeData}}} = _schema(eltype(P))
_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:Tuple} = ((:value,), (eltype(P),))
function _childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:NamedTuple}
    recname = name(fieldtype(fieldtype(T, :meta), :outer_dim))
    schemas = map(_fieldschema, fieldtypes(P))
    _allequal(schemas) || error("TreeArrays Tables adapter: record fields have mismatched schemas ($(fieldnames(P)) -> $(schemas)) -- heterogeneous records are not a supported Tables shape")
    fnames, ftypes = first(schemas)
    (Tuple(vcat([recname], collect(fnames))), Tuple(vcat([Symbol], collect(ftypes))))
end
_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:TreeData} = _schema(P)   # bookkeeping wrapper -> recurse straight through
function _childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P}   # scalar leaf
    P === Missing && error("TreeArrays Tables adapter: an absent-dim `missing` sentinel is not representable as a table row -- unsupported shape")
    ((:value,), (P,))
end

_fieldschema(::Type{F}) where F<:TreeData = _schema(F)
function _fieldschema(::Type{F}) where F
    F === Missing && error("TreeArrays Tables adapter: a record field holding the absent-dim `missing` sentinel is not representable as a table row -- unsupported shape")
    F <: AbstractArray && error("TreeArrays Tables adapter: a raw (non-TreeData) array-valued record field carries no dim labels -- unsupported shape")
    ((:value,), (F,))
end

_allequal(xs) = all(==(first(xs)), xs)

Tables.istable(::Type{<:TreeData}) = true
Tables.columnaccess(::Type{<:TreeData}) = true
Tables.columnnames(X::TreeData) = _schema(typeof(X))[1]
Tables.schema(X::TreeData) = Tables.Schema(_schema(typeof(X))...)

# ---- the melt: one recursive walk building concrete rows, run exactly once
#      per `Tables.columns(X)` call (the export boundary). ----

# a node's own fixed dims (ghosts dropped, axis-without-position an error) as
# constant columns merged into the accumulated row prefix -- mirrors
# `_ownschema`'s bookkeeping/scalar-leaf branch at the value level.
_ownfixed(alldims::Tuple, prefix::NamedTuple) = merge(prefix, NamedTuple(name(d) => meta(d).values for d in _flatfixed(alldims)))

function _meltarray(X::TreeData, p, prefix::NamedTuple, leafcall)
    axdims, fixed = _splitmelt(TreeArrays.dims(X), _nax(typeof(p)))
    base = merge(prefix, NamedTuple(name(d) => meta(d).values for d in fixed))
    rows = Any[]
    for I in _positions(p)
        axcols = NamedTuple(name(d) => _dimvalue(d, Tuple(I)[k]) for (k, d) in enumerate(axdims))
        append!(rows, leafcall(p[I], merge(base, axcols)))
    end
    rows
end

_melt(X::TreeData, prefix::NamedTuple) = _meltnode(X, parent(X), prefix)
_meltnode(X::TreeData, p::AbstractArray{<:TreeData}, prefix) = _meltarray(X, p, prefix, (el, pre) -> _melt(el, pre))
_meltnode(X::TreeData, p::AbstractArray, prefix)              = _meltarray(X, p, prefix, (v, pre) -> [merge(pre, (;value=v))])
_meltnode(X::TreeData, p::Tuple{Vararg{<:TreeData}}, prefix)  = _meltarray(X, p, prefix, (el, pre) -> _melt(el, pre))
_meltnode(X::TreeData, p::Tuple, prefix)                      = _meltarray(X, p, prefix, (v, pre) -> [merge(pre, (;value=v))])
function _meltnode(X::TreeData, p::NamedTuple, prefix)
    rec = outerdim(X)
    base = _ownfixed(filter(d -> name(d) !== name(rec), TreeArrays.dims(X)), prefix)
    rows = Any[]
    for (k, v) in pairs(p)
        append!(rows, _meltfield(v, merge(base, NamedTuple((name(rec) => k,)))))
    end
    rows
end
_meltnode(X::TreeData, p::TreeData, prefix) = _melt(p, _ownfixed(TreeArrays.dims(X), prefix))   # bookkeeping wrapper -> pass straight through
_meltnode(X::TreeData, p, prefix) = [merge(_ownfixed(TreeArrays.dims(X), prefix), (;value=p))]   # scalar leaf

_meltfield(v::TreeData, prefix) = _melt(v, prefix)
_meltfield(v::Missing, prefix) = error("TreeArrays Tables adapter: a record field holding the absent-dim `missing` sentinel is not representable as a table row -- unsupported shape")
_meltfield(v::AbstractArray, prefix) = error("TreeArrays Tables adapter: a raw (non-TreeData) array-valued record field carries no dim labels -- unsupported shape")
_meltfield(v, prefix) = [merge(prefix, (;value=v))]

function _finalizerows(raw::Vector{Any})
    isempty(raw) && return NamedTuple[]
    T = typeof(raw[1])
    all(r -> typeof(r) === T, raw) ||
        error("TreeArrays Tables adapter: melted rows have inconsistent column sets/types -- heterogeneous reduced TreeData is not a supported Tables shape")
    T[raw...]
end

# a genuinely (not just observed-to-be) constant column -- e.g. one built from
# a fixed dim (decision oni1bc) -- is stored as a `Fill` (O(1), concrete
# eltype, `getindex` + fancy-indexing all work) instead of a densified Vector,
# per the AoV contract (avoid needlessly materializing repeated metadata over
# a potentially large row count).
_compact(v::AbstractVector) = isempty(v) || !_allequal(v) ? v : Fill(v[1], length(v))

function Tables.columns(X::TreeData)
    MELT_COUNT[] += 1
    rows = _finalizerows(_melt(X, NamedTuple()))
    map(_compact, Tables.columntable(rows))
end
Tables.getcolumn(X::TreeData, i::Union{Int,Symbol}) = Tables.getcolumn(Tables.columns(X), i)
Tables.rows(X::TreeData) = Tables.rows(Tables.columns(X))
