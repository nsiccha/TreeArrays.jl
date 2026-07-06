# ===================== TreeTable: the lazy tabular VIEW =====================
# A `TreeData` is ND, possibly-ragged data -- NOT itself a table. `TreeTable`
# is the lazy re-presentation that assigns dims to tabular ROLES: the dims
# named in `wide` spread their levels into COLUMNS; every other dim is LONG (a
# row axis). Records (TreeNamedTuple fields) are wide by default -- distinct
# variables are the natural column shape -- so only *axes* you want widened
# need naming. Orientation is a CONSUMER choice imposed by this view, never
# derived from the data (user decision vj192m + refinement 2026-07-06); only
# the wide role is special, the rest defaults to long.
#
# Nothing materializes at construction -- the Tables.jl interface below hands
# back the same lazy view columns as the underlying melt (`tables.jl`),
# oriented per `wide`. The Tables.jl entry points live HERE, on the view, not
# on a bare `TreeData` (which stays ND / non-tabular).
#
# TRANSITION NOTE: the `TreeData` Tables methods in `tables.jl` are kept for
# now so existing consumers (Bruno interim, the benchmark) don't break;
# removing them (so a bare `TreeData` is no longer a Tables source) is the
# final migration step, once the wide-axis pivot lands and consumers move to
# `TreeTable`.

# `W` (a Tuple of Symbols -- the wide dim names) is baked into the type so the
# melt stays type-stable.
struct TreeTable{TX<:TreeData, W}
    x::TX
end

_source(tt::TreeTable) = getfield(tt, :x)
_widedims(::TreeTable{TX,W}) where {TX,W} = W
_widedims(::Type{<:TreeTable{TX,W}}) where {TX,W} = W

_widetuple(w::Symbol)                   = (w,)
_widetuple(w::Tuple{Vararg{Symbol}})    = w
_widetuple(w::AbstractVector{<:Symbol}) = Tuple(w)
_widetuple(::Nothing)                   = ()

# every name in `wide` must be a real (top-level) dim -- fail loudly on a
# typo'd/absent wide-dim, never silently ignore it. (A wide dim nested deeper
# than the top level is a fast-follow, checked when the pivot lands.)
function _validatewide(x::TreeData, w::Tuple{Vararg{Symbol}})
    isempty(w) && return
    have = map(name, TreeArrays.dims(x))
    for nm in w
        nm in have || error("TreeTable: wide=$(nm) is not a dim of this TreeData (dims: $(have))")
    end
end

function TreeTable(x::TreeData; wide=())
    w = _widetuple(wide)
    _validatewide(x, w)
    TreeTable{typeof(x), w}(x)
end

Base.show(io::IO, tt::TreeTable) = print(io, "TreeTable(…; wide=", _widedims(tt), ")")

Tables.istable(::Type{<:TreeTable})      = true
Tables.columnaccess(::Type{<:TreeTable}) = true

# --- wide=() : the default orientation (records-wide, every axis long).
#     Identical to the underlying TreeData melt, which already implements
#     exactly this -- delegate straight to it. (Records-wide is one of the
#     user-sanctioned defaults; an all-long default would be a per-call flag,
#     a fast-follow.)
Tables.columns(tt::TreeTable{TX, ()}) where TX   = _buildcolumns(_source(tt))
Tables.schema(::TreeTable{TX, ()}) where TX      = Tables.Schema(_schema(TX)...)
Tables.columnnames(::TreeTable{TX, ()}) where TX = _schema(TX)[1]

# --- wide=(dims...) : pivot the named AXES' levels into columns. The lazy
#     axis-pivot (each level -> its own `ValueColumn`, named by the level) is
#     the next increment. NOT a silent fallback to long -- fail loudly so no
#     consumer silently gets the wrong shape.
_widewip(tt) = error("TreeTable: wide=$(_widedims(tt)) axis-pivot is not implemented yet (landing next); wide=() (records-wide, axes-long) works now.")
Tables.columns(tt::TreeTable)     = _widewip(tt)
Tables.schema(tt::TreeTable)      = _widewip(tt)
Tables.columnnames(tt::TreeTable) = _widewip(tt)

Tables.getcolumn(tt::TreeTable, i::Union{Int,Symbol}) = Tables.getcolumn(Tables.columns(tt), i)
Tables.rows(tt::TreeTable)                            = Tables.rows(Tables.columns(tt))
