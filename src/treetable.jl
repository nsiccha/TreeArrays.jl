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

# --- wide=(dims...) : spread the named AXES' levels into columns.
#
# The melt already lays every column out over a single shared `rowdims` tuple,
# and every column decodes its row index through that same tuple. So widening a
# set of axes is a pure RE-INDEXING of the columns the long melt already built:
# delete their slots from the row space, and for each combination of their levels
# re-present each value column as a view that pins those slots. Nothing is
# recomputed, nothing is materialized -- `WideColumn` wraps the very same lazy
# `ValueColumn` / `AxisColumn` / `ConstColumn` objects (decision 1uzarfr).
#
# This is why the pivot needs no new melt path and no new leaf walk: one column
# type, reused for all three, at any number of wide dims.
#
# `wide=:band` (one dim) is the case consumers actually want -- AoV's
# `lineribbon(bands=[:lower => :upper])` takes exactly one widened axis, and a
# single dim's combo label is the bare level name. k > 1 multiplies the column
# count by each further axis's level count (and divides the row count by the
# same), which is inherent to a pivot rather than a defect.

# `PLAN` lives in the TYPE so the full-index reconstruction below unrolls and
# constant-folds -- the same reason `_taken`/`_dropn` (tables.jl) exist. For slot `j`
# of the long row space: `PLAN[j] > 0` reads reduced index `PLAN[j]`, `PLAN[j] < 0`
# reads pinned level `-PLAN[j]`. A single wide dim is just `NP == 1`; nothing about
# the re-indexing is special-cased for it.
struct WideColumn{T, C<:AbstractVector{T}, K, KR, NP, PLAN} <: AbstractVector{T}
    col::C                       # a column of the LONG melt, indexed over `rowdims`
    rowdims::NTuple{K,Int}       # the long row space
    redrowdims::NTuple{KR,Int}   # `rowdims` minus every widened slot
    levels::NTuple{NP,Int}       # the level this column pins along each widened slot
    len::Int                     # prod(redrowdims) -- the wide row count
end

# The reduced row space and the slot plan depend only on (rowdims, poss) -- they are the
# SAME for every column of one pivot, so `_pivotcolumns` computes them once and hands
# them down. Only `levels` varies per column. (Recomputing them per column cost ~57% more
# allocation in `Tables.columns`, which the O(structure) acceptance gate caught.)
_deletemany(t::NTuple{K,Int}, poss) where K = Tuple(t[j] for j in 1:K if !(j in poss))
function _slotplan(K::Int, poss::NTuple{NP,Int}) where NP
    plan, r = Vector{Int}(undef, K), 0
    for j in 1:K
        m = findfirst(==(j), poss)
        plan[j] = m === nothing ? (r += 1) : -m
    end
    Tuple(plan)
end

# `PLAN` arrives as a `Val` so it lands in the type without a per-column recomputation.
_widecolumn(col::AbstractVector, rowdims::NTuple{K,Int}, red::NTuple{KR,Int}, ::Val{PLAN},
            levels::NTuple{NP,Int}, len::Int) where {K,KR,PLAN,NP} =
    WideColumn{eltype(col), typeof(col), K, KR, NP, PLAN}(col, rowdims, red, levels, len)
Base.size(c::WideColumn) = (c.len,)
Base.IndexStyle(::Type{<:WideColumn}) = IndexLinear()
function Base.getindex(c::WideColumn{T,C,K,KR,NP,PLAN}, i::Int) where {T,C,K,KR,NP,PLAN}
    ridx = Tuple(CartesianIndices(c.redrowdims)[i])
    full = ntuple(j -> (p = PLAN[j]; p > 0 ? ridx[p] : c.levels[-p]), Val(K))
    c.col[LinearIndices(c.rowdims)[full...]]
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
# A combo's column name joins one level label per widened dim, in `wide` order. With a
# single wide dim that is the bare level label (`:lower`) -- which is what makes the
# output drop into `lineribbon(bands=[:lower => :upper])`.
_combolabel(labels, ls) = Symbol(join((labels[m][ls[m]] for m in eachindex(ls)), '_'))

function _pivotcolumns(tt::TreeTable)
    w = _widedims(tt)
    allunique(w) || error("TreeTable: wide=$(w) names the same dim more than once")
    long = _buildcolumns(_source(tt))
    wcols = map(nm -> long[nm], w)   # `_validatewide` already proved each name is in the melt
    for (nm, c) in zip(w, wcols)
        c isa AxisColumn || error("TreeTable: wide=$(nm) is a fixed/ghost dim, not a real axis -- it has no levels to spread into columns")
    end

    # every melt column decodes through the SAME `rowdims`, so one widened dim's view of
    # it is every widened dim's view of it
    rowdims = first(wcols).rowdims
    poss    = map(c -> c.pos, wcols)
    nlevels = map(p -> rowdims[p], poss)
    labels  = map((nm, c, L) -> ntuple(l -> _levelname(nm, _dimvalue(c.values, l)), L), w, wcols, nlevels)

    # Two DISTINCT levels of one dim can sanitize to the same label -- `Symbol("q0.025")`
    # and `:q0_025` both become `:q0_025`. Checked PER DIM: across dims a repeat is fine,
    # since `_combolabel` joins one label from each (`:lo` under both `:band` and `:arm`
    # gives `lo_lo`). The later `allunique(names)` would also trip on this, but would
    # blame a melt column rather than the dim whose levels actually collided.
    for (nm, labs) in zip(w, labels)
        allunique(labs) || error("TreeTable: wide=$(nm) has levels that collide as column names after sanitizing: $(_dups(labs)) (from $(labs))")
    end

    # k wide dims -> the cartesian product of their levels. Column count multiplies, which
    # is inherent to a pivot (and why no consumer has wanted k > 1); rows divide by the
    # same factor. Nothing densifies either way -- `WideColumn` still wraps the same lazy
    # melt columns.
    combos = Iterators.product(map(L -> 1:L, nlevels)...)
    red  = _deletemany(rowdims, poss)                # shared by every output column
    plan = Val(_slotplan(length(rowdims), poss))
    len  = prod(red; init=1)

    names, cols = Symbol[], Any[]
    for (nm, c) in pairs(long)
        nm in w && continue                          # their levels BECOME the columns
        if c isa ValueColumn
            for ls in combos
                # `:value` is the long melt's placeholder name for an unnamed leaf and
                # carries no information; a record field's name does, and is kept.
                suffix = _combolabel(labels, ls)
                push!(names, nm === :value ? suffix : Symbol(nm, '_', suffix))
                push!(cols, _widecolumn(c, rowdims, red, plan, Tuple(ls), len))
            end
        else
            # an id column (axis coord / fixed value) reads only its OWN slot, which is
            # never a widened one, so the levels it is pinned at cannot change what it
            # returns -- pin level 1 everywhere.
            push!(names, nm)
            push!(cols, _widecolumn(c, rowdims, red, plan, map(_ -> 1, poss), len))
        end
    end
    # Per-dim levels are unique by now, so a duplicate here means one of two other things:
    #   * a level label collides with an ID column (`wide=:band` with a `:lower` level, on
    #     a tree that already has a `:lower` dim) or with another field's prefixed name;
    #   * two level COMBINATIONS join to the same name -- `_` is not an injective separator,
    #     so levels (`:x`, `:x_y`) x (`:y_z`, `:z`) give `x_y_z` twice.
    # `NamedTuple` would catch both, but only as "duplicate field name in NamedTuple",
    # which names neither the pivot nor the culprit.
    allunique(names) || error("TreeTable: wide=$(w) produced duplicate column names $(_dups(names)) -- a level label collides with another column of the melt, or two level combinations join to the same name")
    NamedTuple{Tuple(names)}(Tuple(cols))
end

_dups(names) = unique(nm for nm in names if count(==(nm), names) > 1)

Tables.columns(tt::TreeTable)     = _pivotcolumns(tt)
Tables.columnnames(tt::TreeTable) = keys(_pivotcolumns(tt))
Tables.schema(tt::TreeTable)      = (c = _pivotcolumns(tt); Tables.Schema(keys(c), map(eltype, values(c)); stored=true))

Tables.getcolumn(tt::TreeTable, i::Union{Int,Symbol}) = Tables.getcolumn(Tables.columns(tt), i)
Tables.rows(tt::TreeTable)                            = Tables.rows(Tables.columns(tt))
