# ===================== Tables.jl integration =====================
# A reduced TreeData is a lazy Tables.jl COLUMN source: nothing melts/
# densifies at construction (eager compute / lazy assembly, decision
# 1uzarfr) -- every column handed back by `Tables.columns` is a lazy VIEW
# computed on access (decision from the user's live steering, 2026-07-06:
# TA->AoV must never repeat metadata before the final JSON-values boundary).
# `Tables.columnnames`/`Tables.schema` are computed from the TYPE alone (no
# instance access at all).
#
# REGULAR (rectangular) trees only -- a ragged tree (unequal sibling shapes
# under an outer array/tuple-of-TreeData axis, or across record fields)
# errors clearly at `Tables.columns` (the only point an INSTANCE exists to
# detect it; `Tables.schema` is type-only and cannot see it -- this
# asymmetry is intentional, not a defect). Ragged support is a separate,
# scoped fast-follow (CSR/ragged-offset machinery), not built here.
#
# Supported shape (errors clearly otherwise -- never a silently-wrong table):
# recurses over the SAME structural cases `_schema` already dispatches on:
# TreeRaggedArray (outer array of TreeData -- recurse per representative
# element, once sibling-uniformity is verified), TreeArray / TreeTuple
# (array/tuple-backed leaf -- one row per position, `:value` column),
# TreeNamedTuple (record -- outer_dim's name becomes a column holding the
# field key; every field must melt to an IDENTICAL child schema), and a
# generic `TreeData` fallback (a bookkeeping wrapper around another
# TreeData, or a bare scalar leaf).
#
# Ghost/missing/fixed dim -> column mapping (decision oni1bc):
#   `sliced` ghost (values === nothing)   -> dropped, no coordinate to record
#   unlabelled axis (values === missing)  -> column kept, value = 1-based position
#   fixed scalar (e.g. quantile's scalar p, or a fixed `random_effect=:on` dim)
#                                          -> column kept, constant every row
#   (a stray real axis with no enclosing array/tuple/record position is
#   unsupported -- there's nowhere to read a per-row coordinate from)

using Tables

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

# AbstractArray uses ndims (mirrors `_reduceouter`'s array handling); Tuple
# (a TreeTuple leaf) is a plain 1-dimensional positional sequence.
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

# ---- schema: (names, types) from the TYPE alone, no instance access. ----
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

# ===================== lazy view columns =====================
# Three concrete column types (dispatch by type, never a runtime kind tag).
# Every column is a concrete-eltype `AbstractVector` computed on access --
# nothing is ever assembled/densified ahead of a consumer's own `collect`/
# `rowtable`/JSON-materialization call.

# A fixed dim's constant column -- ignores the row index entirely. The
# degenerate view: O(1) storage regardless of row count.
struct ConstColumn{T} <: AbstractVector{T}
    value::T
    len::Int
end
Base.size(c::ConstColumn) = (c.len,)
Base.getindex(c::ConstColumn, i::Int) = c.value
Base.IndexStyle(::Type{<:ConstColumn}) = IndexLinear()

# A real axis coordinate OR a record's field-key column -- both are "decode
# row i's position along axis slot `pos` (out of the tree-wide shared
# `rowdims`), look up in a small fixed collection". `values` is `missing`
# for an unlabelled (positional) axis, a TreeDim's coordinate collection for
# a real axis, or a record's field-names tuple for the record-key column.
struct AxisColumn{T,D,K} <: AbstractVector{T}
    values::D
    pos::Int
    rowdims::NTuple{K,Int}
    len::Int
end
function AxisColumn(values, pos::Int, rowdims::NTuple{K,Int}, len::Int) where K
    T = values === missing ? Int : eltype(values)
    AxisColumn{T,typeof(values),K}(values, pos, rowdims, len)
end
Base.size(c::AxisColumn) = (c.len,)
function Base.getindex(c::AxisColumn, i::Int)
    idx = Tuple(CartesianIndices(c.rowdims)[i])
    _dimvalue(c.values, idx[c.pos])
end
Base.IndexStyle(::Type{<:AxisColumn}) = IndexLinear()

# The single `:value` column -- holds the ROOT TreeData BY REFERENCE (never
# copied) and, on access, decodes row i's full tree-position and walks down
# to the terminal scalar. Mirrors the same structural dispatch `_schema`
# uses, index-driven instead of type-driven-only.
struct ValueColumn{T,TX,K} <: AbstractVector{T}
    x::TX
    rowdims::NTuple{K,Int}
    len::Int
end
function ValueColumn(x::TreeData, rowdims::NTuple{K,Int}, len::Int) where K
    T = last(_schema(typeof(x))[2])
    ValueColumn{T,typeof(x),K}(x, rowdims, len)
end
Base.size(c::ValueColumn) = (c.len,)
function Base.getindex(c::ValueColumn, i::Int)
    idx = Tuple(CartesianIndices(c.rowdims)[i])
    _valueat(c.x, idx)
end
Base.IndexStyle(::Type{<:ValueColumn}) = IndexLinear()

# ---- compile-time-specialized tuple head/tail (the type-stability fold-in
#      -- a runtime `idx[m+1:end]` range-slice returns a variable-length
#      Tuple and allocates; dropping via `Base.tail` recursion on a `Val`
#      keeps every intermediate tuple's length statically known). ----
_naxval(::Type{<:AbstractArray{<:Any,N}}) where N = Val(N)
_naxval(::Type{<:Tuple}) = Val(1)

_taken(t::Tuple, ::Val{N}) where N = ntuple(k -> t[k], Val(N))
_dropn(t::Tuple, ::Val{0}) = t
_dropn(t::Tuple, ::Val{N}) where N = _dropn(Base.tail(t), Val(N - 1))

# ---- the `:value` walk: consumes idx components from the front, mirroring
#      `_schema`'s structural dispatch exactly (same rules, index-driven). ----
_valueat(X::TreeData, idx::Tuple) = _valueat_node(parent(X), idx)

function _valueat_node(p::AbstractArray{<:TreeData}, idx::Tuple)
    v = _naxval(typeof(p))
    _valueat(p[CartesianIndex(_taken(idx, v))], _dropn(idx, v))
end
function _valueat_node(p::Tuple{Vararg{<:TreeData}}, idx::Tuple)
    _valueat(p[idx[1]], _dropn(idx, Val(1)))
end
_valueat_node(p::AbstractArray, idx::Tuple) = p[CartesianIndex(_taken(idx, _naxval(typeof(p))))]
_valueat_node(p::Tuple, idx::Tuple) = p[idx[1]]
function _valueat_node(p::NamedTuple, idx::Tuple)
    _valueat_field(p[idx[1]], _dropn(idx, Val(1)))
end
_valueat_node(p::TreeData, idx::Tuple) = _valueat(p, idx)   # bookkeeping wrapper -- consumes nothing
_valueat_node(p, idx::Tuple) = p                             # scalar leaf terminal

_valueat_field(v::TreeData, idx::Tuple) = _valueat(v, idx)
_valueat_field(v, idx::Tuple) = v

# ---- rowdims: one integer per row-varying dim (real axes + record-key
#      "axes"), in nesting order (outermost first). Row index i (1-based,
#      1:prod(rowdims)) maps to a per-level position tuple via native
#      `CartesianIndices(rowdims)` -- Julia's own column-major convention,
#      the outermost/first entry varies fastest. Every AxisColumn/ValueColumn
#      of one `Tables.columns` call decodes `i` via the identical rowdims
#      tuple, so all columns agree row-for-row by construction. This is a
#      fresh canonical order (not a preservation of the old melt's implicit
#      order) -- Tables.jl doesn't mandate one, and no test pins one down;
#      value-equivalence tests compare row SETS, not row-for-row order.
#
#      Ragged guard: raggedness can only arise where an array/tuple-of-
#      TreeData or a NamedTuple's fields have genuinely independent instance
#      shapes (a lone plain array/tuple/namedtuple is uniform by
#      construction). At exactly those boundaries, verify every sibling's
#      SIZE signature (`_rowdims` itself, recursively -- sizes/lengths/
#      nfields only, never touching axis coordinate VALUES) agrees before
#      trusting one representative path for the deeper structure/coordinate
#      lookups -- matches the supervisor's own definition of ragged (unequal
#      child LENGTHS). This is a deliberate scoping choice, not an oversight:
#      comparing full coordinate VALUES (catching the narrower "same length,
#      different per-sibling coordinates" case too) was tried and costs
#      O(rows) whenever siblings are themselves the row-terminal (e.g. every
#      per-slice scalar leaf a `quantile`/`mapslices` call produces is,
#      structurally, exactly this shape) -- which breaks the flat/O(1)-
#      construction requirement for the single most common output shape.
#      Sizes-only stays O(structure) always. Flagged to the supervisor as a
#      scope note: a hand-built array of TreeData siblings with the SAME
#      length but genuinely differing axis coordinate values is not
#      detected here (axis/coordinate columns would silently use one
#      sibling's coordinates for all -- `:value` stays correct regardless,
#      it always walks the real tree per row).
_rowdims(X::TreeData) = _rowdims_node(parent(X))

_rowdims_node(p::AbstractArray) = size(p)   # plain array leaf -- always uniform (one array)
_rowdims_node(p::Tuple) = (length(p),)       # plain tuple leaf -- ditto
_rowdims_node(p::TreeData) = _rowdims(p)     # bookkeeping wrapper -- 0 own dims
_rowdims_node(p) = ()                         # scalar leaf terminal

function _rowdims_node(p::AbstractArray{<:TreeData})
    reps = map(_rowdims, p)
    _allequal(reps) ||
        error("TreeArrays Tables adapter: sibling TreeData elements (among $(length(p))) have inconsistent shape -- ragged trees are not a supported Tables shape yet (regular/rectangular only)")
    (size(p)..., first(reps)...)
end
function _rowdims_node(p::Tuple{Vararg{<:TreeData}})
    reps = map(_rowdims, p)
    _allequal(reps) ||
        error("TreeArrays Tables adapter: sibling TreeData elements (among $(length(p))) have inconsistent shape -- ragged trees are not a supported Tables shape yet (regular/rectangular only)")
    (length(p), first(reps)...)
end
function _rowdims_node(p::NamedTuple)
    reps = map(_fielddims, values(p))
    _allequal(reps) ||
        error("TreeArrays Tables adapter: record fields $(keys(p)) have inconsistent shape -- ragged trees are not a supported Tables shape yet (regular/rectangular only)")
    (length(p), first(reps)...)
end
_fielddims(v::TreeData) = _rowdims(v)
_fielddims(v) = ()

# ---- building the columns: mirrors `_ownschema`+`_childschema`'s dispatch
#      exactly, threading (root, rowdims, offset, n) instead of accumulating
#      (names, types) -- `root` is the ORIGINAL TreeData `Tables.columns` was
#      called on (every ValueColumn walks from there); `offset` is how many
#      leading `rowdims` slots enclosing levels have already claimed. ----
function _buildcolumns(X::TreeData)
    rowdims = _rowdims(X)
    n = prod(rowdims; init=1)
    names, cols = _buildnode(X, X, parent(X), rowdims, 0, n)
    NamedTuple{names}(cols)
end

function _buildnode(root::TreeData, X::TreeData, p::Union{AbstractArray,Tuple}, rowdims, offset, n)
    nax = _nax(typeof(p))
    axdims, fixed = _splitmelt(TreeArrays.dims(X), nax)
    axcols = ntuple(k -> AxisColumn(meta(axdims[k]).values, offset + k, rowdims, n), nax)
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    cnames, ccols = _buildchild(root, p, rowdims, offset + nax, n)
    ((map(name, axdims)..., map(name, fixed)..., cnames...), (axcols..., fixcols..., ccols...))
end
function _buildnode(root::TreeData, X::TreeData, p::NamedTuple, rowdims, offset, n)
    recname = name(outerdim(X))
    fixed = _flatfixed(filter(d -> name(d) !== recname, TreeArrays.dims(X)))
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    reccol = AxisColumn(keys(p), offset + 1, rowdims, n)
    fnames, fcols = _buildfield(root, first(values(p)), rowdims, offset + 1, n)
    ((map(name, fixed)..., recname, fnames...), (fixcols..., reccol, fcols...))
end
function _buildnode(root::TreeData, X::TreeData, p, rowdims, offset, n)   # bookkeeping wrapper (p::TreeData) or scalar leaf
    fixed = _flatfixed(TreeArrays.dims(X))
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    vnames, vcols = _terminalbuild(root, p, rowdims, offset, n)
    ((map(name, fixed)..., vnames...), (fixcols..., vcols...))
end

function _buildchild(root::TreeData, p::AbstractArray{<:TreeData}, rowdims, offset, n)
    c = first(p)
    _buildnode(root, c, parent(c), rowdims, offset, n)
end
function _buildchild(root::TreeData, p::Tuple{Vararg{<:TreeData}}, rowdims, offset, n)
    c = first(p)
    _buildnode(root, c, parent(c), rowdims, offset, n)
end
_buildchild(root::TreeData, p::AbstractArray, rowdims, offset, n) = ((:value,), (ValueColumn(root, rowdims, n),))
_buildchild(root::TreeData, p::Tuple, rowdims, offset, n) = ((:value,), (ValueColumn(root, rowdims, n),))

_buildfield(root::TreeData, v::TreeData, rowdims, offset, n) = _buildnode(root, v, parent(v), rowdims, offset, n)
_buildfield(root::TreeData, v, rowdims, offset, n) = ((:value,), (ValueColumn(root, rowdims, n),))

_terminalbuild(root::TreeData, p::TreeData, rowdims, offset, n) = _buildnode(root, p, parent(p), rowdims, offset, n)
_terminalbuild(root::TreeData, p, rowdims, offset, n) = ((:value,), (ValueColumn(root, rowdims, n),))

function Tables.columns(X::TreeData)
    _schema(typeof(X))   # cheap, type-only -- preserves every existing validation
                          # (heterogeneous records, absent-dim missing, raw array field)
    _buildcolumns(X)
end
Tables.getcolumn(X::TreeData, i::Union{Int,Symbol}) = Tables.getcolumn(Tables.columns(X), i)
Tables.rows(X::TreeData) = Tables.rows(Tables.columns(X))
