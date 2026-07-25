# ===================== Tables.jl integration =====================
# A reduced TreeData is a lazy Tables.jl COLUMN source: nothing melts/
# densifies at construction (eager compute / lazy assembly, decision
# 1uzarfr) -- every column handed back by `Tables.columns` is a lazy VIEW
# computed on access (decision from the user's live steering, 2026-07-06:
# TA->AoV must never repeat metadata before the final JSON-values boundary).
# `Tables.columnnames`/`Tables.schema` are computed from the TYPE alone (no
# instance access at all).
#
# RAGGED trees melt LONG (snag `ragged-tree-cann`). Long form needs no
# rectangularity -- each row's coordinate is read from the position it
# occupies in its OWN sub-tree -- so an outer array/tuple-of-TreeData axis
# whose siblings have different row counts, different axis coordinates, or
# different fixed-dim values is a first-class source here. Row count is
# `sum(length, leaves)` rather than a product of axis extents; `_plan`
# expresses that with a CSR offset table (`RaggedPlan`) and the columns
# below such a boundary re-read their dim per row from the sibling the row
# lands in (`WalkAxisColumn`/`WalkConstColumn`). `Tables.schema` is
# unchanged and still type-only: the column names and types of a ragged
# melt are exactly its rectangular counterpart's, which is why it already
# reported them correctly while `Tables.columns` refused.
#
# Two shapes remain unsupported, both because they are genuinely not
# expressible rather than merely unbuilt:
#   * a NON-CONCRETE tree type (siblings whose axis lengths differ AS TYPES
#     -- a `:dose_mg` axis of `(10, 20)` beside one of `(20,)`). The
#     type-only schema walk cannot run at all, so there is no static schema
#     to serve. Caught in `_schema`.
#   * `TreeTable(x; wide=…)` over a ragged tree. A pivot spreads one axis's
#     levels into columns, which needs ONE level set and ONE column length;
#     a ragged axis has neither. Caught in `_pivotcolumns`, and it says so.
# RECORD FIELDS are also still required to share one row shape -- they lay
# out side by side over ONE shared row space by construction (wide emit,
# decision 1kpyu7n), so a differing-shape field is not a raggedness this
# melt can express either.
#
# Supported shape (errors clearly otherwise -- never a silently-wrong table):
# recurses over the SAME structural cases `_schema` already dispatches on:
# TreeRaggedArray (outer array of TreeData -- recurse through ONE
# representative element when the siblings agree, and switch that boundary to
# the CSR/per-row-walk path when they do not), TreeArray / TreeTuple
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
# name/kind -- `_sigsagree`) since they lay out side by side over
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
# There are two flavours of raggedness, and only one of them is an instance property.
# Siblings that differ only in SIZE (`randn(2)` beside `randn(3)`, both `:time`) share a
# type, so the type-level walk below succeeds and `_plan` melts them long on the instance.
# But siblings that differ in axis LENGTH-AS-TYPE -- a `:dose_mg` axis of `(10, 20)` beside
# one of `(20,)` -- have different `TreeDim` types, so `[a, b]` widens to a non-concrete
# eltype and the type walk itself hits the ragged tree. `fieldtype(T, :meta)` is then the
# abstract `NamedTuple` and the next line died with "type NamedTuple has no field dims" --
# an internal error where this adapter promises a clear one everywhere else.
function _schema(::Type{T}) where T<:TreeData
    # `T` non-concrete has exactly two causes, and consumers hit both. Name them, rather
    # than letting the accessors below die with Base's "type NamedTuple has no field dims".
    #   (a) RAGGED: siblings whose axis lengths differ AS TYPES (a `:dose_mg` axis of
    #       `(10, 20)` beside one of `(20,)`) widen `[a, b]` to a non-concrete eltype.
    #       Siblings differing only in SIZE share a type, so THEY melt (long) on the
    #       instance via `_plan`; only this type-level flavour is unsupportable,
    #       because no static schema can be walked out of a non-concrete type.
    #   (b) EMPTY: `TreeData(TreeData[], ...)`, the natural-looking empty container, erases
    #       the child structure this walk reads.
    isconcretetype(T) || error("TreeArrays Tables adapter: `$T` is not a concrete TreeData type. Either sibling TreeData elements have inconsistent TYPES -- ragged trees are not a supported Tables shape yet (regular/rectangular only) -- or an empty container was spelled `TreeData(TreeData[], ...)` instead of carrying its element type (`TreeData(typeof(leafproto)[], :assay_name => String[])`)")
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
#      Assumes, as `_buildwide` does (backed by `_plan`/`_fieldsig`'s
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

# ===================== row plans =====================
# How a flat row index `i` decodes into a per-level position tuple `idx`.
# ONE arity for the whole tree (K row-varying slots, type-determined -- see
# `_zerodims`), but two row-space SHAPES:
#
#   `DensePlan`  -- the row space is a Cartesian PRODUCT of `dims`. This is
#       the rectangular case and it is byte-identical to what this adapter
#       did before ragged support: `CartesianIndices(dims)[i]`, Julia's own
#       column-major order (the outermost/first entry varies FASTEST).
#
#   `RaggedPlan` -- an array/tuple-of-TreeData boundary whose siblings have
#       DIFFERENT row counts, so no product exists. CSR offsets: element `k`
#       owns rows `offsets[k]+1 : offsets[k+1]`, and `plans[k]` decodes
#       within it. Total rows = `last(offsets)` = `sum(length, leaves)`,
#       exactly the row count a long melt of ragged data must have.
#       Order here is outer-SLOWEST (each element's rows are contiguous),
#       the only order a CSR table can serve cheaply. Row ORDER is not part
#       of the Tables.jl contract and no test pins one down (see `_plan`'s
#       note below), so the two shapes are free to differ; what MUST hold --
#       and does, since every column of one `Tables.columns` call decodes
#       through the SAME plan object -- is that all columns agree row-for-row.
#
# Cost: a `RaggedPlan` allocates ONE `Int` per sibling, once per
# `Tables.columns` call. That is O(ragged-axis extent), not O(rows), and it
# is inherent to raggedness rather than an implementation shortcut: with
# per-sibling lengths that differ, random-access row decode needs the offset
# table (the alternative is an O(siblings) scan per element access). Leaf
# DATA is still never touched, copied, or materialized -- the lazy invariant
# the rectangular path holds to O(depth) holds here to O(structure).
struct DensePlan{K}
    dims::NTuple{K,Int}
end
struct RaggedPlan{K,C}
    csize::NTuple{K,Int}     # the ragged container's OWN array size
    offsets::Vector{Int}     # length N+1, cumulative row counts (offsets[1] == 0)
    plans::C                 # one sub-plan per element, in linear order
end

_nrows(p::DensePlan) = prod(p.dims; init=1)
_nrows(p::RaggedPlan) = @inbounds p.offsets[end]

# arity: how many idx slots this plan fills (identical across a RaggedPlan's
# siblings -- guaranteed by the concrete-type precondition `_schema` enforces)
_arity(::DensePlan{K}) where K = K
_arity(p::RaggedPlan{K}) where K = K + _arity(first(p.plans))

_decodeplan(p::DensePlan, i::Int) = Tuple(CartesianIndices(p.dims)[i])
function _decodeplan(p::RaggedPlan, i::Int)
    k = searchsortedlast(p.offsets, i - 1)          # which sibling owns row i
    @inbounds (Tuple(CartesianIndices(p.csize)[k])..., _decodeplan(p.plans[k], i - p.offsets[k])...)
end

# structural equality -- used where sibling/field row spaces must coincide
# (record fields lay out side by side over ONE shared row space).
_planequal(a::DensePlan, b::DensePlan) = a.dims == b.dims
_planequal(a::RaggedPlan, b::RaggedPlan) =
    a.csize == b.csize && a.offsets == b.offsets && all(map(_planequal, a.plans, b.plans))
_planequal(a, b) = false

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
struct AxisColumn{T,D,P} <: AbstractVector{T}
    values::D
    pos::Int
    plan::P
    len::Int
end
function AxisColumn(values, pos::Int, plan, len::Int)
    T = values === missing ? Int : eltype(values)
    AxisColumn{T,typeof(values),typeof(plan)}(values, pos, plan, len)
end
Base.size(c::AxisColumn) = (c.len,)
function Base.getindex(c::AxisColumn, i::Int)
    idx = _decodeplan(c.plan, i)
    _dimvalue(c.values, idx[c.pos])
end
Base.IndexStyle(::Type{<:AxisColumn}) = IndexLinear()

# ---- per-sibling coordinate columns: the ragged counterpart of the two
#      columns above. `AxisColumn`/`ConstColumn` hold ONE `values` object,
#      read once from a representative sibling -- sound only while every
#      sibling agrees on it. Below a DIVERGENT boundary (siblings differing
#      in row count, in axis coordinates, or in a fixed dim's value) there is
#      no such single object, so these decode the row to its full tree
#      position and WALK to the node the row actually lands in, reading THAT
#      node's own dim. `S` (how many idx slots enclosing levels claim before
#      the target node) and `D` (the dim's index in that node's `dims` tuple)
#      are baked into the type at build time, so the walk unrolls statically
#      exactly as `_valueat`'s does.
#
#      This is what makes a ragged melt strictly MORE faithful than the
#      hand-rolled `reduce(vcat, …)` + `fill(label, n)` it replaces: a
#      per-subject fixed dim (`dose = 20` beside `dose = 200` -- metadata
#      stored ONCE per subject, skill §6) is read per row from its OWN
#      subject, never smeared from a representative. Contrast the rectangular
#      path, where cross-sibling fixed-dim agreement stays an ASSUMPTION
#      (decision 4b3vcd's beta case, deliberately left open because checking
#      it there would cost O(rows)); here the boundary is already paying
#      O(siblings) for its offset table, so the full check is free and is
#      made.
struct WalkAxisColumn{T,TX,P,FP,S,D} <: AbstractVector{T}
    x::TX
    plan::P
    len::Int
    fieldpath::FP
end
function WalkAxisColumn{T}(x::TreeData, plan, len::Int, fieldpath::Tuple{Vararg{Symbol}}, ::Val{S}, ::Val{D}) where {T,S,D}
    WalkAxisColumn{T,typeof(x),typeof(plan),typeof(fieldpath),S,D}(x, plan, len, fieldpath)
end
Base.size(c::WalkAxisColumn) = (c.len,)
function Base.getindex(c::WalkAxisColumn{T,TX,P,FP,S,D}, i::Int) where {T,TX,P,FP,S,D}
    idx = _decodeplan(c.plan, i)
    node = _walkto(c.x, idx, c.fieldpath, Val(S))
    # the target node's own axes occupy global idx slots S+1 .. S+nax, in
    # `dims` order (`_splitmelt`'s invariant), so dim D sits at slot S+D.
    _dimvalue(TreeArrays.dims(node)[D], idx[S + D])
end
Base.IndexStyle(::Type{<:WalkAxisColumn}) = IndexLinear()

struct WalkConstColumn{T,TX,P,FP,S,D} <: AbstractVector{T}
    x::TX
    plan::P
    len::Int
    fieldpath::FP
end
function WalkConstColumn{T}(x::TreeData, plan, len::Int, fieldpath::Tuple{Vararg{Symbol}}, ::Val{S}, ::Val{D}) where {T,S,D}
    WalkConstColumn{T,typeof(x),typeof(plan),typeof(fieldpath),S,D}(x, plan, len, fieldpath)
end
Base.size(c::WalkConstColumn) = (c.len,)
function Base.getindex(c::WalkConstColumn{T,TX,P,FP,S,D}, i::Int) where {T,TX,P,FP,S,D}
    idx = _decodeplan(c.plan, i)
    node = _walkto(c.x, idx, c.fieldpath, Val(S))
    meta(TreeArrays.dims(node)[D]).values
end
Base.IndexStyle(::Type{<:WalkConstColumn}) = IndexLinear()

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
struct ValueColumn{T,TX,P,FP<:Tuple{Vararg{Symbol}}} <: AbstractVector{T}
    x::TX
    plan::P
    len::Int
    fieldpath::FP
end
function ValueColumn{T}(x::TreeData, plan, len::Int, fieldpath::Tuple{Vararg{Symbol}}=()) where T
    ValueColumn{T,typeof(x),typeof(plan),typeof(fieldpath)}(x, plan, len, fieldpath)
end
Base.size(c::ValueColumn) = (c.len,)
function Base.getindex(c::ValueColumn, i::Int)
    idx = _decodeplan(c.plan, i)
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

# ---- the PARTIAL walk: `_valueat`'s descent, stopped after `S` idx slots
#      have been consumed -- i.e. AT the node whose own dims a
#      `WalkAxisColumn`/`WalkConstColumn` must read. Same structural
#      dispatch, same `Val`-driven statically-unrolled tuple arithmetic; the
#      only difference is the terminal condition (`Val{0}` -> return the
#      node) instead of bottoming out at a scalar. `S` is a build-time
#      constant, and `_nax(typeof(p))` folds from the type, so `Val(S - …)`
#      stays a compile-time subtraction and the whole walk inlines.
_walkto(X::TreeData, idx::Tuple, fieldpath::Tuple, ::Val{0}) = X
_walkto(X::TreeData, idx::Tuple, fieldpath::Tuple, ::Val{S}) where S = _walkto_node(parent(X), idx, fieldpath, Val(S))

function _walkto_node(p::AbstractArray{<:TreeData}, idx::Tuple, fieldpath::Tuple, ::Val{S}) where S
    v = _naxval(typeof(p))
    _walkto(p[CartesianIndex(_taken(idx, v))], _dropn(idx, v), fieldpath, Val(S - _nax(typeof(p))))
end
_walkto_node(p::Tuple{Vararg{TreeData}}, idx::Tuple, fieldpath::Tuple, ::Val{S}) where S =
    _walkto(p[idx[1]], _dropn(idx, Val(1)), fieldpath, Val(S - 1))
_walkto_node(p::NamedTuple, idx::Tuple, fieldpath::Tuple, ::Val{S}) where S =
    _walkto(p[fieldpath[1]], idx, Base.tail(fieldpath), Val(S))   # a record consumes a fieldpath entry, no idx slot
_walkto_node(p::TreeData, idx::Tuple, fieldpath::Tuple, ::Val{S}) where S = _walkto(p, idx, fieldpath, Val(S))

# ---- zero-row trees: the empty ragged nesting. ----
#      `_plan`, `_coordsig` and `_buildcolumns` each descend through ONE
#      REPRESENTATIVE child (`first(p)`) at every array/tuple-of-TreeData
#      boundary. An EMPTY such boundary -- what a product-mapped sweep must
#      emit for a cell with no data, `TreeData(typeof(leaf)[], :assay => String[])`
#      -- has no representative, so each of those `first` calls threw a bare
#      `BoundsError`. TreeArrays PRODUCES this value itself: reducing the
#      perfectly meltable `TreeData(Float64[], :assay => String[])` over any dim
#      yields an empty ragged array its OWN melt then could not consume.
#
#      An empty node contributes ZERO rows, and zero rows need no coordinates at
#      all -- only column NAMES and TYPES, which `_schema` already derives from
#      the TYPE alone. So the instance walk only has to stay TOTAL and report the
#      right row ARITY: `_zerodims` mirrors `_plan`'s recursion over the
#      element TYPE, emitting a 0 for every row-varying slot. An empty node's
#      child EXTENTS are genuinely unknowable (an array's size is not in its
#      type) and also irrelevant -- `prod(rowdims) == 0` either way, so no row
#      index is ever decoded through them. Only the arity matters, and that IS
#      in the type.
_zerodims(::Type{T}) where T<:TreeData = _zerodims_node(_parentof(T))
_zerodims_node(::Type{P}) where P<:AbstractArray{<:TreeData} = (ntuple(_ -> 0, _nax(P))..., _zerodims(eltype(P))...)
_zerodims_node(::Type{P}) where P<:Tuple{TreeData,Vararg{TreeData}} = (0, _zerodims(eltype(P))...)   # "at least one" -- see `_eltype` (types.jl)
_zerodims_node(::Type{P}) where P<:AbstractArray = ntuple(_ -> 0, _nax(P))
_zerodims_node(::Type{P}) where P<:Tuple = (0,)
_zerodims_node(::Type{P}) where P<:NamedTuple = _zerodims_field(fieldtype(P, 1))   # wide emit: a record adds no row-dim
_zerodims_node(::Type{P}) where P<:TreeData = _zerodims(P)                          # bookkeeping wrapper
_zerodims_node(::Type{P}) where P = ()                                              # scalar leaf terminal
_zerodims_field(::Type{F}) where F<:TreeData = _zerodims(F)
_zerodims_field(::Type{F}) where F = ()

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
#      Divergence check, two parts, both run at every array/tuple-of-
#      TreeData / NamedTuple-fields boundary (a lone plain array/tuple/
#      namedtuple is uniform by construction -- these are the ONLY places
#      sibling instances can genuinely diverge). Neither part is a REFUSAL
#      any more: together they pick this boundary's row plan and decide
#      whether a representative's metadata may stand for every sibling.
#      1. SIZE signature (`_plan` itself, recursively -- sizes/lengths/
#         nfields only) -- catches unequal child LENGTHS, which route the
#         boundary to a `RaggedPlan` (CSR offsets) instead of a `DensePlan`.
#      2. COORDINATE signature (`_coordsig`, below) -- catches the narrower
#         "same length, different per-sibling coordinate VALUES" case, which
#         keeps the dense product row space but clears `shared`, so the
#         columns below read per row (`WalkAxisColumn`) instead of off one
#         representative. `_coordsig` mirrors `_plan`'s own recursive shape
#         -- descending through the SAME representative (`first(p)`) at
#         every nested boundary that `_buildcolumns` itself uses -- but
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
#         `_plan`'s existing recursion reaches THAT nested boundary and
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
# A node's row plan, plus the two things the column builder needs to descend
# without recomputing anything: whether a representative's coordinate/fixed
# metadata may stand for every sibling (`shared`), and the representative
# child's own `PlanNode` (`child`).
struct PlanNode{P,C}
    plan::P
    shared::Bool
    child::C
end

_plan(X::TreeData) = _plan_node(parent(X))

_plan_node(p::AbstractArray) = PlanNode(DensePlan(size(p)), true, nothing)   # plain array leaf -- always uniform (one array)
_plan_node(p::Tuple) = PlanNode(DensePlan((length(p),)), true, nothing)       # plain tuple leaf -- ditto
_plan_node(p::TreeData) = _plan(p)                                            # bookkeeping wrapper -- 0 own dims
_plan_node(p) = PlanNode(DensePlan(()), true, nothing)                        # scalar leaf terminal

function _plan_node(p::AbstractArray{<:TreeData})
    # No representative, and none needed -- `size(p)` already carries a 0 (see
    # `_zerodims`), so `_nrows` is 0 and `_buildcolumns` returns
    # `_emptycolumns` before anything ever reads `.child`.
    #
    # This branch's `child::Nothing` is what makes THIS method's return type a
    # small `Union` (the populated branch's child is a `PlanNode`), which costs
    # one box + one dynamic dispatch per `Tables.columns` call on an
    # array-of-`TreeData` root. Deliberately left: it is O(1) and flat in both
    # sibling count and row count (measured 32 bytes at 1 and at 5000
    # siblings), whereas every way to unify the two branches either invents a
    # representative for an empty container or re-walks the subtree per
    # structural node. The costs the Delta A gate guards -- per-sibling and
    # per-row -- are unaffected.
    isempty(p) && return PlanNode(DensePlan((size(p)..., _zerodims(eltype(p))...)), true, nothing)
    _plan_container(p, size(p))
end
_plan_node(p::Tuple{Vararg{TreeData}}) = _plan_container(p, (length(p),))

# The one place a tree can be ragged. Streams the sibling scan rather than
# `map`ping it into a Vector: the uniform path must stay allocation-flat as
# the sibling count scales (the Delta A acceptance gate scales a record tree
# to 5000 siblings), and a `PlanNode` -- unlike the `()` the old size-only
# signature returned for a scalar leaf -- is not a zero-size type, so
# materializing one per sibling WOULD show up there. The ragged path builds
# its offset table and per-sibling plans, which is O(siblings) by necessity.
function _plan_container(p, csize)
    rep = _plan(first(p))
    sameshape, allshared = true, true
    for x in p
        k = _plan(x)
        if !_planequal(k.plan, rep.plan)
            sameshape = false
            break
        end
        k.shared || (allshared = false)
    end
    if sameshape && rep.plan isa DensePlan
        # The row space IS a product, so keep the cheap dense decode. Whether the
        # COLUMNS below may be shared is a separate question: siblings can agree on
        # shape and still carry their own coordinate values (subject A measured at
        # its times, subject B at different ones) -- that is a perfectly meltable
        # long table, it just cannot read its `:time` column off one representative.
        shared = allshared && _sigsagree(p, _coordsig)
        return PlanNode(DensePlan((csize..., rep.plan.dims...)), shared, rep)
    end
    offsets = Vector{Int}(undef, length(p) + 1)
    offsets[1] = 0
    plans = map(x -> _plan(x).plan, p)
    for (k, pl) in enumerate(plans)
        offsets[k + 1] = offsets[k] + _nrows(pl)
    end
    PlanNode(RaggedPlan(csize, offsets, plans), false, rep)
end

function _plan_node(p::NamedTuple)
    # `all(map(f, kids))`, not `all(f, kids)`: `kids` is a possibly
    # heterogeneous tuple, and only the `map` spelling unrolls (same reason
    # `_sigsagree` splits on `::Tuple` -- this walk runs once per sibling).
    kids = map(_fieldplan, values(p))
    rep = first(kids)
    all(map(k -> _planequal(k.plan, rep.plan), kids)) ||
        error("TreeArrays Tables adapter: record fields $(keys(p)) have inconsistent shape -- a record's fields lay out side by side over ONE shared row space, so fields of differing shape are not representable (regular/rectangular only)")
    shared = all(map(k -> k.shared, kids)) && _sigsagree(values(p), _fieldsig)
    PlanNode(rep.plan, shared, rep)   # wide emit: the record contributes NO row-dim of its own (decision 1kpyu7n) -- fields
                                       # become columns, not extra rows; their (validated-identical) deeper shape is the row-dim.
end
_fieldplan(v::TreeData) = _plan(v)
_fieldplan(v) = PlanNode(DensePlan(()), true, nothing)

# ---- coordinate signature: `_plan`'s recursive shape, collecting
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

# an EMPTY child boundary has no representative to descend through, and no coordinates to
# compare -- a sibling that is empty already differs from a non-empty one in `_plan`'s
# SIZE check (0 vs n), which routes the boundary to a `RaggedPlan` where the empty
# sibling simply contributes zero rows.
_coordsig_child(p::AbstractArray{<:TreeData}) = isempty(p) ? () : _coordsig(first(p))   # ONE representative path down
_coordsig_child(p::Tuple{Vararg{TreeData}}) = isempty(p) ? () : _coordsig(first(p))
_coordsig_child(p::AbstractArray) = ()   # plain array leaf -- no deeper TreeData child
_coordsig_child(p::Tuple) = ()

# ---- field signature: the SAME recursive shape as `_coordsig`, but
#      collecting EVERY own dim (axis AND fixed AND ghost, tagged with its
#      KIND) -- used ONLY at the NamedTuple record-field-agreement boundary
#      (`_plan_node(::NamedTuple)`), where the element count is the
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

_fieldsig_child(p::AbstractArray{<:TreeData}) = isempty(p) ? () : _fieldsig(first(p))   # ONE representative path down (empty -> nothing to compare, see `_coordsig_child`)
_fieldsig_child(p::Tuple{Vararg{TreeData}}) = isempty(p) ? () : _fieldsig(first(p))
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

# Do these siblings/fields agree on the signature `sigfn` collects? A `false`
# is NOT an error any more -- it routes the columns below this boundary to
# the per-sibling `Walk*` reads (see their comment).
#
# TWO iteration shapes, because the two call sites differ in kind:
#
#   * SIBLINGS arrive as an array/tuple of `TreeData` whose elements share a
#     type, so a streamed loop infers concretely AND is the only O(1)-memory
#     option at 5000 siblings -- never `map`ped into a Vector, for the
#     allocation reason `_plan_container` documents.
#   * record FIELDS arrive as `values(::NamedTuple)`, a possibly
#     HETEROGENEOUS tuple. A `for` loop over one cannot unroll, so every
#     element boxes and every `sigfn` call dispatches dynamically. That is
#     paid once per SIBLING when a ragged/uniform container holds records
#     (the Delta A gate is exactly this: 5000 siblings x a two-field
#     `(Float64, Int)` record), and it dominated the whole plan walk at
#     ~112 bytes/sibling. `map` over a tuple IS unrolled, so routing tuples
#     through it keeps that path allocation-flat.
#
# Comparison is whole-signature, not element-wise: `Tuple`'s `isequal` is
# itself recursively unrolled, is length-safe (unequal lengths return `false`
# instead of an out-of-bounds `ref[k]`), and keeps the per-element semantics
# `_sigmatch` documents -- verified directly for the identity fast path, an
# equal-but-distinct copy, `NaN`, `missing` and a length mismatch.
_sigsagree(elems, sigfn) = _sigsagree_streamed(elems, sigfn)
_sigsagree(elems::Tuple, sigfn) = _sigsagree_unrolled(map(sigfn, elems))

function _sigsagree_streamed(elems, sigfn)
    ref = sigfn(first(elems))
    for x in elems
        _sigmatch(sigfn(x), ref) || return false
    end
    true
end

_sigsagree_unrolled(sigs::Tuple) = all(map(s -> _sigmatch(s, first(sigs)), sigs))

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
# The invariant part of a build: the root every `ValueColumn`/`Walk*` column
# walks from, the tree-wide row plan every column decodes through, and the
# row count. Bundled so the recursive builders keep a readable arity once
# `pn` (the current node's `PlanNode`) and `walk` join the thread.
struct BuildCtx{R,P}
    root::R
    plan::P
    n::Int
end

function _buildcolumns(X::TreeData)
    # The type-only validation runs HERE, not at the `Tables.columns(::TreeData)`
    # entry point, because `_buildcolumns` has more than one caller: `TreeTable`
    # with no `wide=` dispatches straight to it, and the `wide=` pivot builds the
    # long columns through it too. While a ragged INSTANCE was refused, `_plan`
    # happened to throw first and masked the gap; now that ragged melts, an
    # unvalidatable tree (a non-concrete/jagged tree type, whose schema cannot be
    # walked from the type at all) would have slipped through the `TreeTable`
    # door and produced columns whose element types are `Any`. One call, cheap
    # and type-only, covers every door.
    _schema(typeof(X))
    pn = _plan(X)
    n = _nrows(pn.plan)
    n == 0 && return _emptycolumns(X)
    names, cols = _buildnode(BuildCtx(X, pn.plan, n), X, parent(X), 0, pn, false)
    NamedTuple{names}(cols)
end

# ---- column emission for ONE node's own dims. `walk` is the sticky "some
#      enclosing boundary's siblings diverge" flag: while it is false these
#      are the O(1)-storage shared columns this adapter has always built;
#      once true, each dim is re-read per row from the sibling the row
#      actually lands in (see `WalkAxisColumn`). `offset` doubles as the
#      Walk* columns' `S` (how many idx slots enclosing levels claim), and a
#      dim's index in the node's own `dims` tuple as their `D`.
_fixedidxs(alldims, keep) = Tuple(k for k in eachindex(alldims) if keep(k) && _dimkind(alldims[k]) === :fixed)

_axiscols(ctx, axdims, offset, walk::Bool, fp) =
    ntuple(k -> walk ?
                WalkAxisColumn{_axistype(typeof(axdims[k]))}(ctx.root, ctx.plan, ctx.n, fp, Val(offset), Val(k)) :
                AxisColumn(meta(axdims[k]).values, offset + k, ctx.plan, ctx.n),
           length(axdims))

_fixedcols(ctx, alldims, fixedidx, offset, walk::Bool, fp) =
    map(k -> walk ?
             WalkConstColumn{_fixedtype(typeof(alldims[k]))}(ctx.root, ctx.plan, ctx.n, fp, Val(offset), Val(k)) :
             ConstColumn(meta(alldims[k]).values, ctx.n),
        fixedidx)

# Zero rows: SOME axis (or ragged nesting) below has length 0, so `_buildnode`'s
# representative walk (`first(p)`, `first(values(fields))`) has nothing to descend into
# at that level. It also has nothing to DO: no row index will ever be decoded. Emit the
# type-derived schema as zero-length, concretely-typed vectors -- a lazy view column
# carries no information at length 0, so this is not an eager densification (there is no
# row DATA to densify), and it keeps `Tables.columns` TOTAL on every tree
# `Tables.schema` accepts. `_schema` is type-only, so the names/types here are the exact
# ones `Tables.schema` reports -- the two cannot drift.
function _emptycolumns(X::TreeData)
    names, types = _schema(typeof(X))
    NamedTuple{names}(map(T -> T[], types))
end

function _buildnode(ctx::BuildCtx, X::TreeData, p::Union{AbstractArray,Tuple}, offset, pn, walk::Bool, fieldpath::Tuple{Vararg{Symbol}}=())
    alldims = TreeArrays.dims(X)
    nax = _nax(typeof(p))
    axdims, fixed = _splitmelt(alldims, nax)
    axcols = _axiscols(ctx, axdims, offset, walk, fieldpath)
    fixidx = _fixedidxs(alldims, k -> k > nax)
    fixcols = _fixedcols(ctx, alldims, fixidx, offset, walk, fieldpath)
    cnames, ccols = _buildchild(ctx, p, offset + nax, pn, walk, fieldpath)
    ((map(name, axdims)..., map(name, fixed)..., cnames...), (axcols..., fixcols..., ccols...))
end
# wide emit (decision 1kpyu7n): every field becomes its OWN column-group,
# name-prefixed, sharing the record's row plan (the record itself contributes
# NO row-dim -- see `_plan_node(::NamedTuple)`). Fields are validated
# consistent (sizes, coordinates, AND dim name/kind -- scope-fork 2) by
# `_plan`/`_sigsagree` BEFORE `_buildcolumns` ever starts, so no
# re-validation happens here; heterogeneous field TYPES are fine (decision
# 1vbt15w) since each field gets its own concretely-typed column(s), never a
# shared/boxed one.
#
# `wfp` is the fieldpath a SHARED column must use if it has to walk: shared
# columns deliberately do not append a field name (that is what makes them
# shared), but `_walkto` still has to pick SOME field to descend through at
# this record. Every field agrees on the dim in question -- that is the
# precondition for sharing it -- so the representative's name is the correct
# and cheapest choice.
function _buildnode(ctx::BuildCtx, X::TreeData, p::NamedTuple, offset, pn, walk::Bool, fieldpath::Tuple{Vararg{Symbol}}=())
    alldims = TreeArrays.dims(X)
    recname = name(outerdim(X))
    fixed = _flatfixed(filter(d -> name(d) !== recname, alldims))
    fixidx = _fixedidxs(alldims, k -> name(alldims[k]) !== recname)
    fixcols = _fixedcols(ctx, alldims, fixidx, offset, walk, fieldpath)
    wfp = (fieldpath..., first(keys(p)))
    fnames, fcols = _buildwide(ctx, p, offset, pn.child, walk || !pn.shared, fieldpath, wfp)
    ((map(name, fixed)..., fnames...), (fixcols..., fcols...))
end
function _buildnode(ctx::BuildCtx, X::TreeData, p, offset, pn, walk::Bool, fieldpath::Tuple{Vararg{Symbol}}=())   # bookkeeping wrapper (p::TreeData) or scalar leaf
    alldims = TreeArrays.dims(X)
    fixed = _flatfixed(alldims)
    fixidx = _fixedidxs(alldims, _ -> true)
    fixcols = _fixedcols(ctx, alldims, fixidx, offset, walk, fieldpath)
    vnames, vcols = _terminalbuild(ctx, p, offset, pn, walk, fieldpath)
    ((map(name, fixed)..., vnames...), (fixcols..., vcols...))
end

# ---- wide fan-out, de-duplicated: fields validated structurally identical
#      (by `_plan`/`_sigsagree`, scope-fork 2 included) share
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
_buildwide(ctx::BuildCtx, fields::NamedTuple, offset, pn, walk::Bool, fieldpath, wfp) =
    _buildwide_dispatch(ctx, fields, first(values(fields)), offset, pn, walk, fieldpath, wfp)

_buildwide_dispatch(ctx::BuildCtx, fields::NamedTuple, rep::TreeData, offset, pn, walk::Bool, fieldpath, wfp) =
    _buildwide_parent(ctx, fields, rep, parent(rep), offset, pn, walk, fieldpath, wfp)
_buildwide_dispatch(ctx::BuildCtx, fields::NamedTuple, rep, offset, pn, walk::Bool, fieldpath, wfp) =   # rep is a plain scalar -- every field is independently a terminal NOW
    _buildfanout(ctx, fields, offset, pn, walk, fieldpath)

function _buildwide_parent(ctx::BuildCtx, fields::NamedTuple, rep::TreeData, p::Union{AbstractArray,Tuple}, offset, pn, walk::Bool, fieldpath, wfp)
    alldims = TreeArrays.dims(rep)
    nax = _nax(typeof(p))
    axdims, fixed = _splitmelt(alldims, nax)
    axcols = _axiscols(ctx, axdims, offset, walk, wfp)
    fixidx = _fixedidxs(alldims, k -> k > nax)
    fixcols = _fixedcols(ctx, alldims, fixidx, offset, walk, wfp)
    cnames, ccols = _buildwidechild(ctx, fields, p, offset + nax, pn, walk, fieldpath, wfp)
    ((map(name, axdims)..., map(name, fixed)..., cnames...), (axcols..., fixcols..., ccols...))
end
_buildwide_parent(ctx::BuildCtx, fields::NamedTuple, rep::TreeData, p::TreeData, offset, pn, walk::Bool, fieldpath, wfp) =   # bookkeeping wrapper -- peel through per field, keep sharing
    _buildwide(ctx, map(parent, fields), offset, pn, walk, fieldpath, wfp)
_buildwide_parent(ctx::BuildCtx, fields::NamedTuple, rep::TreeData, p::NamedTuple, offset, pn, walk::Bool, fieldpath, wfp) =   # a nested record -- fields may genuinely diverge in TYPE here (heterogeneity allowed), fan out fully
    _buildfanout(ctx, fields, offset, pn, walk, fieldpath)
function _buildwide_parent(ctx::BuildCtx, fields::NamedTuple, rep::TreeData, p, offset, pn, walk::Bool, fieldpath, wfp)   # a scalar terminal held by a (possibly fixed-dim-bearing) TreeData -- share the fixed dims once, fan out only the terminal value
    alldims = TreeArrays.dims(rep)
    fixed = _flatfixed(alldims)
    fixidx = _fixedidxs(alldims, _ -> true)
    fixcols = _fixedcols(ctx, alldims, fixidx, offset, walk, wfp)
    vnames, vcols = _buildterminalfanout(ctx, fields, v -> typeof(parent(v)), offset, fieldpath)
    ((map(name, fixed)..., vnames...), (fixcols..., vcols...))
end

_buildwidechild(ctx::BuildCtx, fields::NamedTuple, p::AbstractArray{<:TreeData}, offset, pn, walk::Bool, fieldpath, wfp) =
    _buildwide(ctx, map(v -> first(parent(v)), fields), offset, pn.child, walk || !pn.shared, fieldpath, wfp)
_buildwidechild(ctx::BuildCtx, fields::NamedTuple, p::Tuple{Vararg{TreeData}}, offset, pn, walk::Bool, fieldpath, wfp) =
    _buildwide(ctx, map(v -> first(parent(v)), fields), offset, pn.child, walk || !pn.shared, fieldpath, wfp)
_buildwidechild(ctx::BuildCtx, fields::NamedTuple, p::AbstractArray, offset, pn, walk::Bool, fieldpath, wfp) =
    _buildterminalfanout(ctx, fields, v -> eltype(parent(v)), offset, fieldpath)
_buildwidechild(ctx::BuildCtx, fields::NamedTuple, p::Tuple, offset, pn, walk::Bool, fieldpath, wfp) =
    _buildterminalfanout(ctx, fields, v -> eltype(parent(v)), offset, fieldpath)

# every field bottoms out to a terminal value HERE simultaneously (guaranteed
# by the pre-validated shared-rowdims invariant) -- each field's OWN concrete
# type (via `ttype`, the one place a plain-array-element `eltype(parent(v))`
# and a bare-scalar `typeof(parent(v))` differ) becomes its own `ValueColumn`.
function _buildterminalfanout(ctx::BuildCtx, fields::NamedTuple{names}, ttype, offset, fieldpath) where names
    fname = first(names)
    col = ValueColumn{ttype(fields[fname])}(ctx.root, ctx.plan, ctx.n, (fieldpath..., fname))
    rest = NamedTuple{Base.tail(names)}(Base.tail(values(fields)))
    restnames, restcols = _buildterminalfanout(ctx, rest, ttype, offset, fieldpath)
    ((fname, restnames...), (col, restcols...))
end
_buildterminalfanout(ctx::BuildCtx, fields::NamedTuple{()}, ttype, offset, fieldpath) = ((), ())

# fields have diverged (a terminal value, or a nested record) -- recurse
# field-by-field via `NamedTuple{names}`'s own `names` tuple (field COUNT is
# static, so this is compile-time-specialized recursion, matching
# `_taken`/`_dropn`'s existing tuple-recursion idiom -- no intermediate
# Vector accumulator). Every field independently re-derives whatever OWN
# axis/fixed structure it still has below this point -- correct (per-field
# structure can genuinely differ once fields have diverged, e.g. a nested
# record's own further fields), just no longer de-duplicated -- there is
# nothing left to de-duplicate against once fields disagree in KIND.
function _buildfanout(ctx::BuildCtx, p::NamedTuple{names}, offset, pn, walk::Bool, fieldpath) where names
    fname = first(names)
    # `pn` is this record's node; each field's OWN node is `pn.child` (all
    # fields are `_planequal`, so the representative's stands for every one).
    subnames, subcols = _buildfield(ctx, p[fname], offset, pn.child, walk || !pn.shared, (fieldpath..., fname))
    rest = NamedTuple{Base.tail(names)}(Base.tail(values(p)))
    restnames, restcols = _buildfanout(ctx, rest, offset, pn, walk, fieldpath)
    ((map(nm -> _prefixname(fname, nm), subnames)..., restnames...), (subcols..., restcols...))
end
_buildfanout(ctx::BuildCtx, p::NamedTuple{()}, offset, pn, walk::Bool, fieldpath) = ((), ())

# Descending a child boundary is where `walk` turns on: `pn.shared` is false
# exactly when this node's siblings disagree (row count, axis coordinates, or
# a fixed dim's value), so nothing below may be read off a representative.
function _buildchild(ctx::BuildCtx, p::AbstractArray{<:TreeData}, offset, pn, walk::Bool, fieldpath)
    c = first(p)
    _buildnode(ctx, c, parent(c), offset, pn.child, walk || !pn.shared, fieldpath)
end
function _buildchild(ctx::BuildCtx, p::Tuple{Vararg{TreeData}}, offset, pn, walk::Bool, fieldpath)
    c = first(p)
    _buildnode(ctx, c, parent(c), offset, pn.child, walk || !pn.shared, fieldpath)
end
_buildchild(ctx::BuildCtx, p::AbstractArray, offset, pn, walk::Bool, fieldpath) = ((:value,), (ValueColumn{eltype(p)}(ctx.root, ctx.plan, ctx.n, fieldpath),))
_buildchild(ctx::BuildCtx, p::Tuple, offset, pn, walk::Bool, fieldpath) = ((:value,), (ValueColumn{eltype(p)}(ctx.root, ctx.plan, ctx.n, fieldpath),))

_buildfield(ctx::BuildCtx, v::TreeData, offset, pn, walk::Bool, fieldpath) = _buildnode(ctx, v, parent(v), offset, pn, walk, fieldpath)
_buildfield(ctx::BuildCtx, v, offset, pn, walk::Bool, fieldpath) = ((:value,), (ValueColumn{typeof(v)}(ctx.root, ctx.plan, ctx.n, fieldpath),))

_terminalbuild(ctx::BuildCtx, p::TreeData, offset, pn, walk::Bool, fieldpath) = _buildnode(ctx, p, parent(p), offset, pn, walk, fieldpath)
_terminalbuild(ctx::BuildCtx, p, offset, pn, walk::Bool, fieldpath) = ((:value,), (ValueColumn{typeof(p)}(ctx.root, ctx.plan, ctx.n, fieldpath),))

Tables.columns(X::TreeData) = _buildcolumns(X)   # validation lives in `_buildcolumns` -- see there
Tables.getcolumn(X::TreeData, i::Union{Int,Symbol}) = Tables.getcolumn(Tables.columns(X), i)
Tables.rows(X::TreeData) = Tables.rows(Tables.columns(X))
