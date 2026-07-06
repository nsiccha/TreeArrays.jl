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
# (array/tuple-backed leaf -- one row per position, `:value` column), and a
# generic `TreeData` fallback (a bookkeeping wrapper around another
# TreeData, or a bare scalar leaf).
#
# TreeNamedTuple (record) is WIDE, not long (decision 1kpyu7n, user-directed
# 2026-07-06 -- "why pivot to long at all?" for an inherently wide-sliceable
# rectangular reduction): each field becomes its OWN column-group,
# name-prefixed (nesting composes: `outer_inner_...`), sharing the SAME row
# index as its siblings -- the record itself contributes NO row-dim (no
# field-key column, unlike the earlier long-mode design this replaces).
# Fields may have heterogeneous TYPES (decision 1vbt15w) -- each gets its
# own concretely-typed `ValueColumn`, never a shared/boxed one -- but MUST
# share identical deeper row-shape (sizes, coordinate values, AND dim
# name/kind -- `_checksiblingcoords`) since they lay out side by side over
# one shared row space; a scalar field beside an axis-bearing field
# (differing row-shape entirely) is an explicit non-goal, not attempted.
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
_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:Tuple{Vararg{TreeData}}} = _schema(eltype(P))
_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:Tuple} = ((:value,), (eltype(P),))
_childschema(::Type{T}, ::Type{P}) where {T<:TreeData,P<:NamedTuple} = _wideschema(P)
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

# wide-emit column naming (decision 1kpyu7n): a field's terminal `:value`
# becomes just the field's own name; a field that itself melts to MULTIPLE
# sub-columns (a nested record, or a multi-valued leaf) prefixes each with
# the field name -- so nesting composes into `outer_inner` naturally, with
# no bound on depth. No uniformity requirement across fields anymore
# (dropped, along with it, the old "heterogeneous records unsupported"
# restriction -- that existed only because LONG mode funneled every field
# into ONE shared `:value` column, which needed one shared type; wide mode
# gives every field its own column, so nothing requires it -- user decision
# 1vbt15w, relax).
_prefixname(fname::Symbol, sub::Symbol) = sub === :value ? fname : Symbol(fname, :_, sub)

_alldimsof(::Type{T}) where T<:TreeData = Tuple(fieldtype(fieldtype(T, :meta), :dims).parameters)
_parentof(::Type{T}) where T<:TreeData = fieldtype(T, :parent)

# ---- TYPE-level mirror of `_buildwide` (see its comment for the full
#      rationale) -- same de-duplication (shared axis/fixed columns from a
#      representative field's TYPE; fan out only at a terminal value or a
#      nested record), so `Tables.schema` (type-only) and `Tables.columns`
#      (instance-based) always report the IDENTICAL column set. `ftypes` is
#      a NamedTuple whose VALUES are field TYPES (`Type` objects, not
#      instances) -- the type-level analogue of `_buildwide`'s runtime
#      `fields::NamedTuple` of current-node values.
#      Assumes, as `_buildwide` does (backed by `_rowdims`/`_fieldsig`'s
#      RUNTIME checks), that fields share the same STRUCTURAL kind at each
#      shared level -- a same-size-but-different-container-kind mismatch
#      (e.g. one field Tuple-backed, another Array-backed, coincidentally
#      both length 3) isn't distinguishable from TYPES alone; this is an
#      accepted non-goal, the same footing as the existing
#      regular/rectangular-only ragged scope.
function _wideschema(::Type{P}) where P<:NamedTuple
    ftypes = NamedTuple{fieldnames(P)}(fieldtypes(P))
    _wideschema_dispatch(first(values(ftypes)), ftypes)
end

function _wideschema_dispatch(::Type{F}, ftypes) where F<:TreeData
    # a field NOT even TreeData-wrapped here (a raw array, `missing`, or any
    # other divergent shape) can't safely share the representative's axis
    # walk -- bypass sharing for the WHOLE record and let `_schemafanout`
    # process every field independently via `_fieldschema`, which raises
    # the correct explicit errors (raw array / absent-dim `missing`) instead
    # of a confusing internal MethodError from blindly unwrapping a
    # non-TreeData type.
    all(FT -> FT <: TreeData, values(ftypes)) || return _schemafanout(ftypes)
    _wideschema_parent(_parentof(F), F, ftypes)
end
_wideschema_dispatch(::Type{F}, ftypes) where F = _schemafanout(ftypes)   # rep is a plain (non-TreeData) field type -- every field independently a terminal NOW

function _wideschema_parent(::Type{Prep}, ::Type{F}, ftypes) where {Prep<:Union{AbstractArray,Tuple},F<:TreeData}
    axdims, fixed = _splitmelt(_alldimsof(F), _nax(Prep))
    cnames, ctypes = _wideschemachild(Prep, ftypes)
    (Tuple(vcat(collect(map(name, axdims)), collect(map(name, fixed)), collect(cnames))),
     Tuple(vcat(collect(map(_axistype, axdims)), collect(map(_fixedtype, fixed)), collect(ctypes))))
end
_wideschema_parent(::Type{Prep}, ::Type{F}, ftypes) where {Prep<:TreeData,F<:TreeData} =   # bookkeeping wrapper -- peel through per field, keep sharing
    _wideschema_dispatch(Prep, NamedTuple{keys(ftypes)}(map(_parentof, values(ftypes))))
_wideschema_parent(::Type{Prep}, ::Type{F}, ftypes) where {Prep<:NamedTuple,F<:TreeData} =   # nested record -- fields may diverge in TYPE here (heterogeneity allowed), fan out fully
    _schemafanout(ftypes)
function _wideschema_parent(::Type{Prep}, ::Type{F}, ftypes) where {Prep,F<:TreeData}   # a scalar terminal held by a (possibly fixed-dim-bearing) TreeData
    fixed = _flatfixed(_alldimsof(F))
    vnames, vtypes = _schematerminalfanout(ftypes, FT -> _parentof(FT))
    (Tuple(vcat(collect(map(name, fixed)), collect(vnames))), Tuple(vcat(collect(map(_fixedtype, fixed)), collect(vtypes))))
end

_wideschemachild(::Type{Prep}, ftypes) where Prep<:AbstractArray{<:TreeData} =
    _wideschema_dispatch(eltype(Prep), NamedTuple{keys(ftypes)}(map(FT -> eltype(_parentof(FT)), values(ftypes))))
_wideschemachild(::Type{Prep}, ftypes) where Prep<:Tuple{Vararg{TreeData}} =
    _wideschema_dispatch(eltype(Prep), NamedTuple{keys(ftypes)}(map(FT -> eltype(_parentof(FT)), values(ftypes))))
_wideschemachild(::Type{Prep}, ftypes) where Prep<:AbstractArray =
    _schematerminalfanout(ftypes, FT -> eltype(_parentof(FT)))
_wideschemachild(::Type{Prep}, ftypes) where Prep<:Tuple =
    _schematerminalfanout(ftypes, FT -> eltype(_parentof(FT)))

# every field bottoms out to a terminal value HERE simultaneously (assumed,
# per the STRUCTURAL-kind caveat above) -- `extractor` gets each field's OWN
# concrete terminal type (the one place a plain-array-element `eltype(...)`
# and a bare-scalar direct type differ).
function _schematerminalfanout(ftypes::NamedTuple{names}, extractor) where names
    fname = first(names)
    T = extractor(ftypes[fname])
    rest = NamedTuple{Base.tail(names)}(Base.tail(values(ftypes)))
    restnames, resttypes = _schematerminalfanout(rest, extractor)
    ((fname, restnames...), (T, resttypes...))
end
_schematerminalfanout(ftypes::NamedTuple{()}, extractor) = ((), ())

# fields have diverged (a terminal value, or a nested record) -- fan out
# field-by-field via `_fieldschema` (which already recurses fully for a
# TreeData field, or returns a single named `:value` type otherwise),
# prefixing each field's own sub-names.
function _schemafanout(ftypes::NamedTuple{names}) where names
    fname = first(names)
    subnames, subtypes = _fieldschema(ftypes[fname])
    rest = NamedTuple{Base.tail(names)}(Base.tail(values(ftypes)))
    restnames, resttypes = _schemafanout(rest)
    ((map(nm -> _prefixname(fname, nm), subnames)..., restnames...), (subtypes..., resttypes...))
end
_schemafanout(ftypes::NamedTuple{()}) = ((), ())

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

# The single value column -- holds the ROOT TreeData BY REFERENCE (never
# copied) and, on access, decodes row i's full tree-position and walks down
# to the terminal scalar. Mirrors the same structural dispatch `_schema`
# uses, index-driven instead of type-driven-only. `fieldpath` is a FIXED,
# per-column, compile-time-length tuple of record field-keys, baked in at
# construction -- wide-emit's one genuinely new mechanism (decision
# 1kpyu7n): since each named field becomes its OWN column, its ValueColumn
# must always walk that SAME field, not whichever one a row's `idx` might
# otherwise pick. Long-mode leaves (no record ancestor) use the default
# empty fieldpath -- zero behavior change there, `_valueat` never consumes
# it. `T` is supplied explicitly by the caller (the concrete type of THIS
# field's own terminal value, already known from the build-time recursion)
# -- heterogeneous record fields each get their own concretely-typed column
# this way, never a shared/boxed type.
struct ValueColumn{T,TX,K,FP<:Tuple{Vararg{Symbol}}} <: AbstractVector{T}
    x::TX
    rowdims::NTuple{K,Int}
    len::Int
    fieldpath::FP
end
function ValueColumn{T}(x::TreeData, rowdims::NTuple{K,Int}, len::Int, fieldpath::Tuple{Vararg{Symbol}}=()) where {T,K}
    ValueColumn{T,typeof(x),K,typeof(fieldpath)}(x, rowdims, len, fieldpath)
end
Base.size(c::ValueColumn) = (c.len,)
function Base.getindex(c::ValueColumn, i::Int)
    idx = Tuple(CartesianIndices(c.rowdims)[i])
    _valueat(c.x, idx, c.fieldpath)
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

# ---- the value walk: consumes idx components from the front (unnamed
#      axes) and fieldpath components from the front (named/record
#      boundaries) INDEPENDENTLY -- a record consumes ONE fieldpath symbol
#      and ZERO idx slots (wide emit: it contributes no row-dim), every
#      other node passes fieldpath through unchanged and consumes idx as
#      before. Mirrors `_schema`'s structural dispatch exactly (same rules,
#      index-driven for unnamed structure, fieldpath-driven for named).
_valueat(X::TreeData, idx::Tuple, fieldpath::Tuple{Vararg{Symbol}}=()) = _valueat_node(parent(X), idx, fieldpath)

function _valueat_node(p::AbstractArray{<:TreeData}, idx::Tuple, fieldpath::Tuple)
    v = _naxval(typeof(p))
    _valueat(p[CartesianIndex(_taken(idx, v))], _dropn(idx, v), fieldpath)
end
function _valueat_node(p::Tuple{Vararg{TreeData}}, idx::Tuple, fieldpath::Tuple)
    _valueat(p[idx[1]], _dropn(idx, Val(1)), fieldpath)
end
_valueat_node(p::AbstractArray, idx::Tuple, fieldpath::Tuple) = p[CartesianIndex(_taken(idx, _naxval(typeof(p))))]
_valueat_node(p::Tuple, idx::Tuple, fieldpath::Tuple) = p[idx[1]]
function _valueat_node(p::NamedTuple, idx::Tuple, fieldpath::Tuple)
    _valueat_field(p[fieldpath[1]], idx, Base.tail(fieldpath))   # fieldpath-driven field pick -- idx untouched
end
_valueat_node(p::TreeData, idx::Tuple, fieldpath::Tuple) = _valueat(p, idx, fieldpath)   # bookkeeping wrapper -- consumes nothing
_valueat_node(p, idx::Tuple, fieldpath::Tuple) = p                                        # scalar leaf terminal

_valueat_field(v::TreeData, idx::Tuple, fieldpath::Tuple) = _valueat(v, idx, fieldpath)
_valueat_field(v, idx::Tuple, fieldpath::Tuple) = v

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
#      Ragged guard, two parts, both required at every array/tuple-of-
#      TreeData / NamedTuple-fields boundary (a lone plain array/tuple/
#      namedtuple is uniform by construction -- these are the ONLY places
#      sibling instances can genuinely diverge):
#      1. SIZE signature (`_rowdims` itself, recursively -- sizes/lengths/
#         nfields only) -- catches unequal child LENGTHS.
#      2. COORDINATE signature (`_coordsig`, below) -- catches the narrower
#         "same length, different per-sibling coordinate VALUES" case (user-
#         greenlit fast-follow, 2026-07-06, overriding an earlier document-
#         and-defer call). `_coordsig` mirrors `_rowdims`'s own recursive
#         shape -- descending through the SAME representative (`first(p)`)
#         at every nested boundary that `_buildcolumns` itself uses -- but
#         collects `name => meta(d).values` pairs instead of sizes, at every
#         level, not just the boundary's own. That full-depth walk is what
#         closes a real gap a shallower (own-level-only) version has: two
#         siblings can each be internally uniform yet disagree on a DEEPER
#         axis (e.g. subject A's visits all share one `:time` array, subject
#         B's visits all share a DIFFERENT one) -- comparing only each
#         sibling's own-level axis misses this; comparing the full
#         recursive signature catches it, because the representative-path
#         walk for sibling A and for sibling B pass through their own
#         deeper levels directly against each other. Internal
#         inconsistency WITHIN one sibling's own deeper structure (A's
#         visit 1 vs visit 2) is still caught independently, when
#         `_rowdims`'s existing recursion reaches THAT nested boundary and
#         this same two-part check fires there too -- no separate pass
#         needed, it falls out of the existing per-level fold-in.
#      Never touching leaf DATA (only axis/record-key METADATA, and only
#      along one representative path per level, never iterating a full
#      sibling array) keeps this O(structure depth), not O(rows) -- the
#      distinction that matters: an EARLIER attempt at full coordinate
#      checking compared full recursive dims signatures INCLUDING descending
#      into and iterating actual per-slice leaf VALUES, which cost O(rows)
#      for the extremely common "flat array of per-slice scalar leaves"
#      shape (every `quantile`/`mapslices` output). This version never does
#      that -- a scalar leaf / non-TreeData value contributes `()`, so the
#      cost is bounded by tree depth times sibling count at each level, the
#      same order the size check already pays.
#      `_sigmatch` compares each coordinate pair via `===` first (O(1),
#      handles the common case of a hoisted/shared coordinate object, and is
#      `missing`-safe/short-circuiting for unlabelled axes) and falls back to
#      `isequal` (not `==`) only on an identity miss -- `isequal` avoids
#      `==`'s three-valued `missing`-propagation and its `NaN != NaN`
#      surprise, so an independently-constructed-but-equal coordinate array
#      is correctly accepted, not spuriously rejected.
_rowdims(X::TreeData) = _rowdims_node(parent(X))

_rowdims_node(p::AbstractArray) = size(p)   # plain array leaf -- always uniform (one array)
_rowdims_node(p::Tuple) = (length(p),)       # plain tuple leaf -- ditto
_rowdims_node(p::TreeData) = _rowdims(p)     # bookkeeping wrapper -- 0 own dims
_rowdims_node(p) = ()                         # scalar leaf terminal

function _rowdims_node(p::AbstractArray{<:TreeData})
    reps = map(_rowdims, p)
    _allequal(reps) ||
        error("TreeArrays Tables adapter: sibling TreeData elements (among $(length(p))) have inconsistent shape -- ragged trees are not a supported Tables shape yet (regular/rectangular only)")
    _checksiblingcoords(p, _coordsig)
    (size(p)..., first(reps)...)
end
function _rowdims_node(p::Tuple{Vararg{TreeData}})
    reps = map(_rowdims, p)
    _allequal(reps) ||
        error("TreeArrays Tables adapter: sibling TreeData elements (among $(length(p))) have inconsistent shape -- ragged trees are not a supported Tables shape yet (regular/rectangular only)")
    _checksiblingcoords(p, _coordsig)
    (length(p), first(reps)...)
end
function _rowdims_node(p::NamedTuple)
    reps = map(_fielddims, values(p))
    _allequal(reps) ||
        error("TreeArrays Tables adapter: record fields $(keys(p)) have inconsistent shape -- ragged trees are not a supported Tables shape yet (regular/rectangular only)")
    _checksiblingcoords(values(p), _fieldsig)
    first(reps)   # wide emit: the record contributes NO row-dim of its own (decision 1kpyu7n) -- fields
                   # become columns, not extra rows; their (validated-identical) deeper shape is the row-dim.
end
_fielddims(v::TreeData) = _rowdims(v)
_fielddims(v) = ()

# ---- coordinate signature: `_rowdims`'s recursive shape, collecting
#      `name => values` pairs instead of sizes (see the guard comment
#      above). AXIS dims only -- used at the array-of-TreeData /
#      tuple-of-TreeData sibling-check boundary, where the element count is
#      the ROW count and so MUST stay O(structure), not O(rows): a fixed dim
#      contributes nothing here on purpose, because for the common
#      "scalar-leaf-per-row" shape (every `quantile`/`mapslices` output) that
#      keeps `_coordsig` returning the trivial `()` -- a zero-size singleton
#      tuple `map`s over `elems` for free, regardless of row count. Widening
#      this to capture fixed dims too (which `_fieldsig` below does, and an
#      earlier version of this function did) turns that `()` into a small
#      but NON-empty per-element struct, and `map` over N ROWS of those
#      allocates O(rows) -- measured directly: reintroducing it made
#      `Tables.columns` on a 5000-column tree allocate ~25x a 50-column one
#      instead of flat (violates the "no allocated vector anywhere" gate).
#      Splicing (`...`) flattens every level into ONE flat tuple of pairs,
#      so two siblings' signatures compare elementwise regardless of depth.
#
#      Boundary contract (decision 4b3vcd): array/tuple TreeData SIBLINGS are
#      only checked on their AXIS coordinates here -- they are ASSUMED to
#      agree on FIXED-dim values too (a `ConstColumn` is built once, from the
#      representative sibling's value, and never cross-checked against the
#      others). A sibling that genuinely diverges on a fixed value is not
#      detected and silently reports the representative's value for every
#      row. This is a deliberate, permanent boundary, not a gap pending an
#      extension: the natural way to express a value that legitimately
#      varies per sibling is an AXIS coordinate, not a per-element fixed
#      dim: extending this check to fixed dims would need a per-element (not
#      per-structure) walk, reintroducing the O(rows) allocation documented
#      above. Contrast with record FIELDS (`_fieldsig` below), where the
#      element count is field-count-bounded (small, fixed) rather than
#      row-count-bounded, so the fuller name+kind+value check is affordable
#      there and fully closes the analogous gap for record fields (4b3vcd's
#      alpha case; this comment documents its beta case, left open).
_coordsig(X::TreeData) = _coordsig_node(X, parent(X))
_coordsig(x) = ()   # a non-TreeData record-field value (plain scalar) -- no coords to collect

function _coordsig_node(X::TreeData, p::Union{AbstractArray,Tuple})
    axdims, _ = _splitmelt(TreeArrays.dims(X), _nax(typeof(p)))
    (map(d -> name(d) => meta(d).values, axdims)..., _coordsig_child(p)...)
end
_coordsig_node(X::TreeData, p::TreeData) = _coordsig(p)   # bookkeeping wrapper -- 0 own coords
_coordsig_node(X::TreeData, p) = ()                        # scalar leaf terminal (incl. NamedTuple -- not reachable here, only `_fieldsig` walks records)

_coordsig_child(p::AbstractArray{<:TreeData}) = _coordsig(first(p))   # ONE representative path down
_coordsig_child(p::Tuple{Vararg{TreeData}}) = _coordsig(first(p))
_coordsig_child(p::AbstractArray) = ()   # plain array leaf -- no deeper TreeData child
_coordsig_child(p::Tuple) = ()

# ---- field signature: the SAME recursive shape as `_coordsig`, but
#      collecting EVERY own dim (axis AND fixed AND ghost, tagged with its
#      KIND) -- used ONLY at the NamedTuple record-field-agreement boundary
#      (`_rowdims_node(::NamedTuple)`), where the element count is the
#      record's FIELD count, not the row count -- a small, structure-fixed
#      number (2-5 typically), so the same O(N) `map` that would be
#      dangerous for `_coordsig`'s row-bounded call site is safe here. This
#      is what makes the representative-based shared-column build sound for
#      wide-emit record fields (scope-fork 2, mandatory): two fields can
#      match on sizes and on axis VALUES yet still disagree on a FIXED dim's
#      name/value, or on a dim's KIND entirely -- comparing only axis values
#      misses that; comparing the full own-dim set catches it.
_fieldsig(X::TreeData) = (_fieldsig_own(X)..., _fieldsig_child(parent(X))...)
_fieldsig(x) = ()   # a non-TreeData record-field value (plain scalar) -- no coords to collect

_fieldsig_own(X::TreeData) = map(d -> name(d) => (_dimkind(d), meta(d).values), TreeArrays.dims(X))

_fieldsig_child(p::AbstractArray{<:TreeData}) = _fieldsig(first(p))   # ONE representative path down
_fieldsig_child(p::Tuple{Vararg{TreeData}}) = _fieldsig(first(p))
_fieldsig_child(p::AbstractArray) = ()   # plain array leaf -- no deeper TreeData child
_fieldsig_child(p::Tuple) = ()
_fieldsig_child(p::NamedTuple) = _fieldsig(first(values(p)))
_fieldsig_child(p::TreeData) = _fieldsig(p)   # bookkeeping wrapper -- recurse straight through
_fieldsig_child(p) = ()                        # scalar leaf terminal

# `===` on a freshly-built `name => values` (or `name => (kind, values)`)
# pair still hits the O(1) fast path when `values` is a shared/hoisted
# object: `Pair`/`Tuple` are immutable, so Julia's `===` (egal) on them
# recurses structurally field-by-field rather than requiring literal
# same-allocation identity -- verified directly (a fresh `:t => shared` pair
# `===` another fresh one wrapping the SAME `shared` array, `false` for a
# distinct-but-equal copy).
_sigmatch(a, b) = a === b || isequal(a, b)

function _checksiblingcoords(elems, sigfn)
    sigs = map(sigfn, elems)
    ref = first(sigs)
    for s in sigs
        for k in eachindex(s)
            _sigmatch(s[k], ref[k]) ||
                error("TreeArrays Tables adapter: sibling TreeData elements (among $(length(elems))) disagree on axis `$(first(s[k]))`'s coordinate values -- ragged trees are not a supported Tables shape yet (regular/rectangular only)")
        end
    end
end

# ---- building the columns: mirrors `_ownschema`+`_childschema`'s dispatch
#      exactly, threading (root, rowdims, offset, n, fieldpath) instead of
#      accumulating (names, types) -- `root` is the ORIGINAL TreeData
#      `Tables.columns` was called on (every ValueColumn walks from there);
#      `offset` is how many leading `rowdims` slots enclosing levels have
#      already claimed; `fieldpath` is which record field(s) enclosing
#      levels have already committed to (empty outside any record). Every
#      column returned is a lazy view struct (`ConstColumn`/`AxisColumn`/
#      `ValueColumn`) built in O(structure depth) -- no Vector is ever
#      allocated to hold row DATA here; materialization happens only at a
#      consumer's own `rowtable`/`columntable`/`collect` call. ----
function _buildcolumns(X::TreeData)
    rowdims = _rowdims(X)
    n = prod(rowdims; init=1)
    names, cols = _buildnode(X, X, parent(X), rowdims, 0, n)
    NamedTuple{names}(cols)
end

function _buildnode(root::TreeData, X::TreeData, p::Union{AbstractArray,Tuple}, rowdims, offset, n, fieldpath::Tuple{Vararg{Symbol}}=())
    nax = _nax(typeof(p))
    axdims, fixed = _splitmelt(TreeArrays.dims(X), nax)
    axcols = ntuple(k -> AxisColumn(meta(axdims[k]).values, offset + k, rowdims, n), nax)
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    cnames, ccols = _buildchild(root, p, rowdims, offset + nax, n, fieldpath)
    ((map(name, axdims)..., map(name, fixed)..., cnames...), (axcols..., fixcols..., ccols...))
end
# wide emit (decision 1kpyu7n): every field becomes its OWN column-group,
# name-prefixed, sharing the record's rowdims (the record itself contributes
# NO row-dim -- see `_rowdims_node(::NamedTuple)`). Fields are validated
# consistent (sizes, coordinates, AND now dim name/kind -- scope-fork 2) by
# `_rowdims`/`_checksiblingcoords` BEFORE `_buildcolumns` ever starts, so no
# re-validation happens here; heterogeneous field TYPES are fine (decision
# 1vbt15w) since each field gets its own concretely-typed column(s), never a
# shared/boxed one.
function _buildnode(root::TreeData, X::TreeData, p::NamedTuple, rowdims, offset, n, fieldpath::Tuple{Vararg{Symbol}}=())
    recname = name(outerdim(X))
    fixed = _flatfixed(filter(d -> name(d) !== recname, TreeArrays.dims(X)))
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    fnames, fcols = _buildwide(root, p, rowdims, offset, n, fieldpath)
    ((map(name, fixed)..., fnames...), (fixcols..., fcols...))
end
function _buildnode(root::TreeData, X::TreeData, p, rowdims, offset, n, fieldpath::Tuple{Vararg{Symbol}}=())   # bookkeeping wrapper (p::TreeData) or scalar leaf
    fixed = _flatfixed(TreeArrays.dims(X))
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    vnames, vcols = _terminalbuild(root, p, rowdims, offset, n, fieldpath)
    ((map(name, fixed)..., vnames...), (fixcols..., vcols...))
end

# ---- wide fan-out, de-duplicated: fields validated structurally identical
#      (by `_rowdims`/`_checksiblingcoords`, scope-fork 2 included) share
#      their axis/fixed columns -- built ONCE from a representative field,
#      not once per field -- and per-field divergence (prefixed columns)
#      only starts at the point fields ACTUALLY differ: a terminal value
#      (own concrete type per field) or a nested record (own fan-out per
#      field). Without this, N fields walking the SAME deeper axis would
#      each independently re-emit that axis as N byte-identical columns
#      (e.g. 5 stat fields sharing one population/posterior axis pair would
#      otherwise produce 5 redundant `<field>_population` columns instead of
#      one shared `population`). `fields` threads the CURRENT node for every
#      sibling field in parallel with `rep` (any one of them, used only to
#      decide dispatch/read shared dims -- by the pre-validated invariant
#      every field would give the identical answer here).
_buildwide(root::TreeData, fields::NamedTuple, rowdims, offset, n, fieldpath) =
    _buildwide_dispatch(root, fields, first(values(fields)), rowdims, offset, n, fieldpath)

_buildwide_dispatch(root::TreeData, fields::NamedTuple, rep::TreeData, rowdims, offset, n, fieldpath) =
    _buildwide_parent(root, fields, rep, parent(rep), rowdims, offset, n, fieldpath)
_buildwide_dispatch(root::TreeData, fields::NamedTuple, rep, rowdims, offset, n, fieldpath) =   # rep is a plain scalar -- every field is independently a terminal NOW
    _buildfanout(root, fields, rowdims, offset, n, fieldpath)

function _buildwide_parent(root::TreeData, fields::NamedTuple, rep::TreeData, p::Union{AbstractArray,Tuple}, rowdims, offset, n, fieldpath)
    nax = _nax(typeof(p))
    axdims, fixed = _splitmelt(TreeArrays.dims(rep), nax)
    axcols = ntuple(k -> AxisColumn(meta(axdims[k]).values, offset + k, rowdims, n), nax)
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    cnames, ccols = _buildwidechild(root, fields, p, rowdims, offset + nax, n, fieldpath)
    ((map(name, axdims)..., map(name, fixed)..., cnames...), (axcols..., fixcols..., ccols...))
end
_buildwide_parent(root::TreeData, fields::NamedTuple, rep::TreeData, p::TreeData, rowdims, offset, n, fieldpath) =   # bookkeeping wrapper -- peel through per field, keep sharing
    _buildwide(root, map(parent, fields), rowdims, offset, n, fieldpath)
_buildwide_parent(root::TreeData, fields::NamedTuple, rep::TreeData, p::NamedTuple, rowdims, offset, n, fieldpath) =   # a nested record -- fields may genuinely diverge in TYPE here (heterogeneity allowed), fan out fully
    _buildfanout(root, fields, rowdims, offset, n, fieldpath)
function _buildwide_parent(root::TreeData, fields::NamedTuple, rep::TreeData, p, rowdims, offset, n, fieldpath)   # a scalar terminal held by a (possibly fixed-dim-bearing) TreeData -- share the fixed dims once, fan out only the terminal value
    fixed = _flatfixed(TreeArrays.dims(rep))
    fixcols = map(d -> ConstColumn(meta(d).values, n), fixed)
    vnames, vcols = _buildterminalfanout(root, fields, v -> typeof(parent(v)), rowdims, offset, n, fieldpath)
    ((map(name, fixed)..., vnames...), (fixcols..., vcols...))
end

_buildwidechild(root::TreeData, fields::NamedTuple, p::AbstractArray{<:TreeData}, rowdims, offset, n, fieldpath) =
    _buildwide(root, map(v -> first(parent(v)), fields), rowdims, offset, n, fieldpath)
_buildwidechild(root::TreeData, fields::NamedTuple, p::Tuple{Vararg{TreeData}}, rowdims, offset, n, fieldpath) =
    _buildwide(root, map(v -> first(parent(v)), fields), rowdims, offset, n, fieldpath)
_buildwidechild(root::TreeData, fields::NamedTuple, p::AbstractArray, rowdims, offset, n, fieldpath) =
    _buildterminalfanout(root, fields, v -> eltype(parent(v)), rowdims, offset, n, fieldpath)
_buildwidechild(root::TreeData, fields::NamedTuple, p::Tuple, rowdims, offset, n, fieldpath) =
    _buildterminalfanout(root, fields, v -> eltype(parent(v)), rowdims, offset, n, fieldpath)

# every field bottoms out to a terminal value HERE simultaneously (guaranteed
# by the pre-validated shared-rowdims invariant) -- each field's OWN concrete
# type (via `ttype`, the one place a plain-array-element `eltype(parent(v))`
# and a bare-scalar `typeof(parent(v))` differ) becomes its own `ValueColumn`.
function _buildterminalfanout(root::TreeData, fields::NamedTuple{names}, ttype, rowdims, offset, n, fieldpath) where names
    fname = first(names)
    col = ValueColumn{ttype(fields[fname])}(root, rowdims, n, (fieldpath..., fname))
    rest = NamedTuple{Base.tail(names)}(Base.tail(values(fields)))
    restnames, restcols = _buildterminalfanout(root, rest, ttype, rowdims, offset, n, fieldpath)
    ((fname, restnames...), (col, restcols...))
end
_buildterminalfanout(root::TreeData, fields::NamedTuple{()}, ttype, rowdims, offset, n, fieldpath) = ((), ())

# fields have diverged (a terminal value, or a nested record) -- recurse
# field-by-field via `NamedTuple{names}`'s own `names` tuple (field COUNT is
# static, so this is compile-time-specialized recursion, matching
# `_taken`/`_dropn`'s existing tuple-recursion idiom -- no intermediate
# Vector accumulator). Every field independently re-derives whatever OWN
# axis/fixed structure it still has below this point -- correct (per-field
# structure can genuinely differ once fields have diverged, e.g. a nested
# record's own further fields), just no longer de-duplicated -- there is
# nothing left to de-duplicate against once fields disagree in KIND.
function _buildfanout(root::TreeData, p::NamedTuple{names}, rowdims, offset, n, fieldpath) where names
    fname = first(names)
    subnames, subcols = _buildfield(root, p[fname], rowdims, offset, n, (fieldpath..., fname))
    rest = NamedTuple{Base.tail(names)}(Base.tail(values(p)))
    restnames, restcols = _buildfanout(root, rest, rowdims, offset, n, fieldpath)
    ((map(nm -> _prefixname(fname, nm), subnames)..., restnames...), (subcols..., restcols...))
end
_buildfanout(root::TreeData, p::NamedTuple{()}, rowdims, offset, n, fieldpath) = ((), ())

function _buildchild(root::TreeData, p::AbstractArray{<:TreeData}, rowdims, offset, n, fieldpath)
    c = first(p)
    _buildnode(root, c, parent(c), rowdims, offset, n, fieldpath)
end
function _buildchild(root::TreeData, p::Tuple{Vararg{TreeData}}, rowdims, offset, n, fieldpath)
    c = first(p)
    _buildnode(root, c, parent(c), rowdims, offset, n, fieldpath)
end
_buildchild(root::TreeData, p::AbstractArray, rowdims, offset, n, fieldpath) = ((:value,), (ValueColumn{eltype(p)}(root, rowdims, n, fieldpath),))
_buildchild(root::TreeData, p::Tuple, rowdims, offset, n, fieldpath) = ((:value,), (ValueColumn{eltype(p)}(root, rowdims, n, fieldpath),))

_buildfield(root::TreeData, v::TreeData, rowdims, offset, n, fieldpath) = _buildnode(root, v, parent(v), rowdims, offset, n, fieldpath)
_buildfield(root::TreeData, v, rowdims, offset, n, fieldpath) = ((:value,), (ValueColumn{typeof(v)}(root, rowdims, n, fieldpath),))

_terminalbuild(root::TreeData, p::TreeData, rowdims, offset, n, fieldpath) = _buildnode(root, p, parent(p), rowdims, offset, n, fieldpath)
_terminalbuild(root::TreeData, p, rowdims, offset, n, fieldpath) = ((:value,), (ValueColumn{typeof(p)}(root, rowdims, n, fieldpath),))

function Tables.columns(X::TreeData)
    _schema(typeof(X))   # cheap, type-only -- preserves every existing validation
                          # (heterogeneous records, absent-dim missing, raw array field)
    _buildcolumns(X)
end
Tables.getcolumn(X::TreeData, i::Union{Int,Symbol}) = Tables.getcolumn(Tables.columns(X), i)
Tables.rows(X::TreeData) = Tables.rows(Tables.columns(X))
