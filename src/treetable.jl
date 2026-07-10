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

# every name in `wide` must name a real axis somewhere in the melt -- fail loudly
# on a typo'd/absent wide-dim, never silently ignore it. Checked against
# `_schema`, which enumerates every melt column from the TYPE alone, so a dim
# nested BELOW the top level (the band axis a chained reduction leaves on the
# leaf -- the whole point of `wide=:band`) validates too, and no instance is
# touched. Whether that column is a real *axis* (and so has levels to spread)
# needs the instance, so `_pivotcolumns` re-checks it there.
function _validatewide(x::TreeData, w::Tuple{Vararg{Symbol}})
    isempty(w) && return
    have = _schema(typeof(x))[1]
    for nm in w
        nm in have || error("TreeTable: wide=$(nm) is not a dim of this TreeData (melt columns: $(have))")
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

# --- wide=(dim,) : spread ONE axis's levels into columns.
#
# The melt already lays every column out over a single shared `rowdims` tuple,
# and every column decodes its row index through that same tuple. So widening an
# axis is a pure RE-INDEXING of the columns the long melt already built: delete
# the wide axis's slot from the row space, and for each of its levels re-present
# each value column as a view that pins that slot. Nothing is recomputed, nothing
# is materialized -- `WideColumn` wraps the very same lazy `ValueColumn` /
# `AxisColumn` / `ConstColumn` objects (decision 1uzarfr).
#
# This is why the pivot needs no new melt path and no new leaf walk: one column
# type, reused for all three.

# `pos` lives in the TYPE so the tuple splice below stays allocation-free -- the
# same reason `_taken`/`_dropn` (tables.jl) exist.
struct WideColumn{T, C<:AbstractVector{T}, K, KR, POS} <: AbstractVector{T}
    col::C                       # a column of the LONG melt, indexed over `rowdims`
    rowdims::NTuple{K,Int}       # the long row space
    redrowdims::NTuple{KR,Int}   # `rowdims` minus the widened slot
    level::Int                   # the position this column pins along that slot
    len::Int                     # prod(redrowdims) -- the wide row count
end
_deleteat(t::Tuple, pos::Int) = (t[1:pos-1]..., t[pos+1:end]...)   # construction-time only
_insertat(t::Tuple, ::Val{P}, v) where P = (_taken(t, Val(P-1))..., v, _dropn(t, Val(P-1))...)

function WideColumn(col::AbstractVector, rowdims::NTuple{K,Int}, pos::Int, level::Int) where K
    red = _deleteat(rowdims, pos)
    WideColumn{eltype(col), typeof(col), K, K-1, pos}(col, rowdims, red, level, prod(red; init=1))
end
Base.size(c::WideColumn) = (c.len,)
Base.IndexStyle(::Type{<:WideColumn}) = IndexLinear()
function Base.getindex(c::WideColumn{T,C,K,KR,POS}, i::Int) where {T,C,K,KR,POS}
    ridx = Tuple(CartesianIndices(c.redrowdims)[i])
    c.col[LinearIndices(c.rowdims)[_insertat(ridx, Val(POS), c.level)...]]
end

# Vega-Lite reads a dot in a field name as nested property access (aov-use §9),
# so a level like `0.025` must not reach a column name verbatim. This applies to
# EVERY level, Symbol ones included: `quantile(X, :band => (var"q0.025"=0.025, …))`
# and `TreeDim(:band, Symbol.(["0.025", "0.975"]))` both put a dot in a Symbol, and
# a `q0.025` column reads as `datum["q0"]["025"]` in VL -- a silent wrong-data plot,
# never an error. Sanitizing only the non-Symbol path left that hole open on the
# reducer's OWN primary path.
_sanitize(v) = replace(string(v), '.' => '_')
# A Symbol level (what the `:band => spec` reducer produces) is already a good
# column name and is used bare -- that is exactly what makes the output drop into
# `lineribbon(bands=[:lower => :upper])`. Anything else is prefixed by its dim, so
# a widened `time` axis yields `time_0_1`, never a bare `0_1`.
_levelname(::Symbol, v::Symbol) = Symbol(_sanitize(v))
_levelname(wname::Symbol, v) = Symbol(wname, '_', _sanitize(v))

# WIDE-MODE SCHEMA IS A RUNTIME SCHEMA, BY DESIGN (decision 1krjg6l, resolved).
#
# Wide-mode column names ARE the widened axis's coordinate values -- they live in
# `meta(d).values` and are not recoverable from the type. So unlike the long melt,
# `Tables.columns`/`schema` here read the instance and are not inferable in their
# NAMES. That is fine, and Tables.jl says so itself: `schema` is documented to
# return `Union{Nothing, Tables.Schema}`, and `Schema{nothing,nothing}` stores its
# names in a plain `Vector{Symbol}` field precisely because "encoding the names/
# types as type parameters becomes prohibitive to the compiler" for wide tables
# (Tables.jl `src/Tables.jl:454-464`). Type-level names were never a Tables.jl
# requirement -- only a property the LONG melt gets for free, since there the names
# are dim names, which ARE type parameters.
#
# Hence `stored=true`: baking runtime-derived names into a type parameter would
# construct a fresh type per label set for a schema nothing can specialize on
# anyway (the call is not inferable). Measured on a widened 2000-level axis:
# 189 µs / 81 KB typed vs 7 µs / 32 KB stored. Note `sch.names` is therefore a
# `Vector{Symbol}` in wide mode and a `Tuple` in long mode -- both are valid
# `Tables.Schema`s and `.names` reads through either.
#
# The columns themselves stay concretely-typed lazy views, which is what a Tables
# consumer's contract actually requires (aov-use §2).
function _pivotcolumns(tt::TreeTable)
    w = _widedims(tt)
    length(w) == 1 || error("TreeTable: wide=$(w) -- exactly one wide dim is supported (got $(length(w)))")
    wname = only(w)
    long = _buildcolumns(_source(tt))
    wcol = long[wname]   # `_validatewide` already proved the name exists in the melt
    wcol isa AxisColumn || error("TreeTable: wide=$(wname) is a fixed/ghost dim, not a real axis -- it has no levels to spread into columns")

    rowdims, pos = wcol.rowdims, wcol.pos
    nlevels = rowdims[pos]
    labels = ntuple(l -> _levelname(wname, _dimvalue(wcol.values, l)), nlevels)
    allunique(labels) || error("TreeTable: wide=$(wname) levels collide as column names after sanitizing: $(labels)")

    names, cols = Symbol[], Any[]
    for (nm, c) in pairs(long)
        nm === wname && continue                     # its levels BECOME the columns
        if c isa ValueColumn
            for l in 1:nlevels
                # `:value` is the long melt's placeholder name for an unnamed leaf and
                # carries no information; a record field's name does, and is kept.
                push!(names, nm === :value ? labels[l] : Symbol(nm, '_', labels[l]))
                push!(cols, WideColumn(c, rowdims, pos, l))
            end
        else
            # an id column (axis coord / fixed value) reads its own slot only, so the
            # level it is pinned at cannot change what it returns -- pin level 1.
            push!(names, nm)
            push!(cols, WideColumn(c, rowdims, pos, 1))
        end
    end
    # A level label can also collide with an ID column (`wide=:band` with a `:lower`
    # level, on a tree that already has a `:lower` dim) or with another field's
    # prefixed name. `NamedTuple` would catch it, but only as "duplicate field name
    # in NamedTuple" -- which names neither the pivot nor the culprit.
    allunique(names) || error("TreeTable: wide=$(wname) produced duplicate column names $(_dups(names)) -- a level label collides with another column of the melt")
    NamedTuple{Tuple(names)}(Tuple(cols))
end

_dups(names) = unique(nm for nm in names if count(==(nm), names) > 1)

Tables.columns(tt::TreeTable)     = _pivotcolumns(tt)
Tables.columnnames(tt::TreeTable) = keys(_pivotcolumns(tt))
Tables.schema(tt::TreeTable)      = (c = _pivotcolumns(tt); Tables.Schema(keys(c), map(eltype, values(c)); stored=true))

Tables.getcolumn(tt::TreeTable, i::Union{Int,Symbol}) = Tables.getcolumn(Tables.columns(tt), i)
Tables.rows(tt::TreeTable)                            = Tables.rows(Tables.columns(tt))
