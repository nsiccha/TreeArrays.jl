# ===================== mapslices =====================
# Reduce the named `dims`: apply `f` to each leftover-index slice, keep everything.
# `f` returns a TreeData (or a scalar). Reduced dims stay but become `sliced` (aggregated).
#
# `dims=` is foundALL (decision 1iy1r57): every requested name must exist somewhere in the tree
# (a typo throws -- from the TYPE where that proves it, falling back to an instance walk on a
# jagged tree, never standing down), and every branch reached must carry it (a heterogeneous-dims
# shape throws). The `missing` sentinel this used to return for an absent reduce-dim is RETIRED.
# Note `missing` still marks an UNLABELLED AXIS in `TreeDim(:draw)` -- an unrelated job that
# keeps the name (types.jl, dim_helpers.jl).

# Type-stable partition of `alldims` against the (type-level) reduce-dim set `want`, given
# the axis/ghost boundary `nax` (= ndims(parent(X))). Plain recursive tuple-peeling proved
# NOT to fold here once the axis/ghost boundary is actually crossed (verified empirically --
# 3+ element mixed axis/ghost tuples widen to a Union under Julia's inference; a uniform
# all-axis tuple folds fine by coincidence, which is why #1's original all-axis smoke test
# passed before this was checked against a ghost-bearing input like a chained reduction's
# result). `@generated` sidesteps recursion-depth inference limits entirely -- this is a
# SECOND genuine `@generated` spot beyond `_eachslice`, discovered empirically, not reflexive
# (flagged to the supervisor). Mirrors the original redaxes/keepaxes/keptdims/keptset/
# trailing semantics exactly: an axis dim named in `want` sets `found`; an axis dim not in
# `want` is kept; everything else (ghost dims, and axis dims being reduced) lands in
# `trailing`, sliced iff in `want`.
# shared staging-time (not @generated itself) iteration idiom for the `@generated` dims-
# tuple partitioners below: (position, element-type) pairs over a Tuple TYPE's parameters.
# A single combined N-way-bucket-with-transform combinator was considered (per review) and
# rejected -- `_splitdims`'s 4-way/`sliced()`-transform/`foundany`-flag shape, `_splitrecord`'s
# 3-way (a dim can be excluded from BOTH buckets), and `_exclude`'s plain 1-way filter don't
# share enough to justify passing a staged classifier-function through the @generated
# boundary; this loop-shape extraction is the honest amount of dedup without contorting any
# of the three into a worse shared abstraction.
_dimtypes(alldims::Type{<:Tuple}) = ((i, alldims.parameters[i]) for i in 1:length(alldims.parameters))

@generated function _splitdims(alldims::Tuple, ::Val{nax}, ::Val{want}) where {nax,want}
    keepidx = Int[]
    kept    = Expr[]
    trail   = Expr[]
    foundany = false
    for (i, d) in _dimtypes(alldims)
        isax   = i <= nax
        inwant = name(d) in want
        if isax && inwant
            push!(trail, :(sliced(alldims[$i])))
            foundany = true
        elseif isax
            push!(keepidx, i)
            push!(kept, :(alldims[$i]))
        elseif inwant
            push!(trail, :(sliced(alldims[$i])))
        else
            push!(trail, :(alldims[$i]))
        end
    end
    :(($(Tuple(keepidx)), ($(kept...),), ($(trail...),), $foundany))
end

# The one genuinely runtime-dims spot: eachslice(A; dims) needs the axis VALUES (not just
# their count) baked into a type parameter for `Slices`'s type to stay concrete. The Val
# lift (`ks` becomes `where`-bound) is enough for Base's own eachslice to see a literal.
@inline _eachslice(A::AbstractArray, ::Val{ks}) where ks = eachslice(A; dims=ks)

# ===================== coordinate-aware kernels (`coords=true`) =====================
# A kernel sees the DATA slice by default. `mapslices(f, X; dims, coords=true)` additionally
# hands it the COORDINATES of the axis being reduced -- what `trapz(t, y)` and `t[argmax(y)]`
# (AUC and tmax, the two standard non-compartmental PK summaries) need, and what the positional
# workarounds (`baseline = v[1]`, `Δ = v .- v[1]`) provably cannot express. Nothing here needs
# rectangularity: on a RAGGED axis each sub-tree already carries its own coordinate vector, and
# the binding below happens PER NODE, so every subject's kernel call sees ITS OWN grid -- which
# is also why "close over a shared constant" was never an answer for the ragged case.
#
# Opt-in is EXPLICIT here, never arity-sniffed. `hasmethod(f, (slice, coords))` is TRUE for
# generic Base functions that mean something else entirely with two arguments -- `maximum(f,
# itr)` exists, so an arity probe would answer `mapslices(maximum, X; dims=:time)` by calling
# `maximum(y, t)`, silently treating the data slice as a predicate. `@kernel` (kernel.jl) DOES
# infer the opt-in, safely, because it reads the literal argument list at macro-expansion time
# instead of guessing from a value.
#
# Mechanically the REQUEST travels as the kernel: `_NeedsCoords` marks it at the public entry
# and rides the whole recursion untouched -- including the ragged descent, which is precisely
# what gives each sub-tree its own coordinates -- and every node that actually reduces swaps it
# for a `_WithCoords` closure carrying THAT node's coordinates. So `_leafreduce`'s base case
# stays `f(sl)`: the gather loop, `_assemble` and the plain 1-arg path are unchanged and pay
# nothing (no flag threaded through the hot path, no branch at the leaf).
struct _NeedsCoords{F}
    f::F
end
struct _WithCoords{F,C}
    f::F
    coords::C
end
(g::_WithCoords)(sl) = g.f(sl, g.coords)

_wantcoords(f, ::Val{false}) = f
_wantcoords(f, ::Val{true})  = _NeedsCoords(f)

# The reduced AXIS dims at this node, in AXIS order -- which is exactly the order of the slice's
# own dimensions (Base's `Slices` preserves the order of the sliced dims), so the coordinate
# vectors line up positionally with what the kernel receives. `@generated` for the same reason
# as `_splitdims`/`_splitrecord`/`_exclude`: a mixed axis/ghost dims tuple does not fold under
# plain recursive peeling (see the note above `_splitdims`).
@generated function _reducedaxes(alldims::Tuple, ::Val{nax}, ::Val{want}) where {nax,want}
    keep = Expr[]
    for (i, d) in _dimtypes(alldims)
        i <= nax && name(d) in want && push!(keep, :(alldims[$i]))
    end
    :(($(keep...),))
end

# An UNLABELLED axis (`values === missing`, §1) has no coordinates to hand over. Passing
# `missing` through would be the retired absent-dim sentinel in a new costume: the kernel
# would compute `trapz(missing, y)` and return a plausible-looking `missing`. Fail by name,
# with the spelling that works -- same discipline and phrasing as `_sweepvalues` (setdim.jl)
# and `selectdim`'s unlabelled-axis guard.
_coordsof(d::TreeDim) = _coordsof(name(d), meta(d).values)
_coordsof(n::Symbol, ::Missing) = error(
    "TreeArrays: `coords=true` cannot hand the kernel coordinates for dim `$n` -- it is unlabelled " *
    "(values === missing), so the axis has no coordinates. Give them at construction " *
    "(`TreeData(x, :$n => ts)`), or drop `coords=true` and reduce positionally."
)
_coordsof(::Symbol, values) = values

# ONE reduced axis -> the BARE coordinate vector, so the kernel is just `f(y, t)` (the
# overwhelmingly common shape, and exactly what `trapz`/`argmax` want). SEVERAL -> one vector
# per reduced axis, in slice-dim order: the slice is genuinely N-dimensional there, and a
# single flat vector could only lie about which coordinate belongs to which position.
_slicecoords(red::Tuple{<:TreeDim}) = _coordsof(red[1])
_slicecoords(red::Tuple)            = map(_coordsof, red)

_bindcoords(f, alldims, valnax, valwant) = f      # plain kernel: nothing to bind, no cost
_bindcoords(f::_NeedsCoords, alldims, valnax, valwant) =
    _WithCoords(f.f, _slicecoords(_reducedaxes(alldims, valnax, valwant)))

# A POOLED straddle concatenates every leaf's slice into one bag (`_pooledstraddle` below), so
# no single axis's coordinate vector lines up with it -- a pooled (draw x chain) bag has draw-
# AND-chain positions, not one coordinate per element. Refuse by name rather than invent an
# alignment (16fwcnx / 1iy1r57: never answer with a plausible-looking value).
_refusecoords(f, want) = nothing
_refusecoords(::_NeedsCoords, want) = throw(ArgumentError(
    "cannot hand a kernel axis coordinates (`coords=true`) while POOLING dims " * string(want) *
    " across a ragged boundary: the pooled bag concatenates every leaf's slice, so no single axis's " *
    "coordinate vector lines up with it. Reduce the coordinate-bearing dim on its own first " *
    "(`coords=true` works there), then pool the result."))

# Shared "found the axis" bookkeeping for TreeArray/TreeRaggedArray: which parent-array
# positions are being reduced vs kept, and the ghost dims left behind for the reduced ones.
# Returns `nothing` when `want` doesn't hit a real axis here -> the caller decides what that
# means (a true leaf -> sentinel/idempotent re-slice; an intermediate node -> recurse deeper
# into each element).
function _reduceouter(f, X, valwant::Val{want}) where want
    alldims = TreeArrays.dims(X)
    n_ax    = ndims(parent(X))
    @assert all(_isaxis, alldims[1:n_ax]) "_reduceouter assumes the first $n_ax dims of $(typeof(X)) are exactly the parent array's axes, positionally, in order; got a non-axis dim at position $(findfirst(!_isaxis, alldims[1:n_ax])) (dims = $(map(name, alldims)))"
    keepaxes, keptdims, trailing, foundany = _splitdims(alldims, Val(n_ax), valwant)
    foundany || return nothing
    # AFTER `foundany`: a branch that reduces nothing here must not bind (and must not demand
    # coordinates from) an axis it isn't touching -- the ragged recursion re-enters per element.
    g = _bindcoords(f, alldims, Val(n_ax), valwant)
    outs = isempty(keepaxes) ? _leafreduce(g, parent(X)) : map(sl -> _leafreduce(g, sl), _eachslice(parent(X), Val(keepaxes)))
    _assemble(outs, keptdims, trailing)
end

# The gather-reduction: reduce an OUTER axis of a nested result by pushing it down to the
# leaves (pure index arithmetic on `sl`, a gathered slice along that axis -- nothing
# materialized) and recursing. Record (NamedTuple- or Tuple-keyed) -> recurse per key.
# Array-backed leaf -> recurse per position, building a NEW nested array (never
# flattened/stacked -- a chained reduction stays a tree of arrays all the way down).
# Scalar-backed leaf -> no inner positions to preserve; the outer reduction replaces the
# leaf outright (mirrors the plain-array base case below). Plain array leaf -> apply the
# kernel directly; this base case is also what a dense TreeArray reduction needs, so
# `_reduceouter` routes all shapes through `_leafreduce` uniformly.

# TreeNamedTuple (named fields) and TreeTuple (positional fields) are both "keyed, axis-
# bearing" records and walk the identical gather-by-key / recurse / reassemble shape; only
# how keys are read and how the container is rebuilt differs -- shared driver below, two
# tiny per-container dispatch points instead of a bespoke tuple-only branch.
_reassemble(::NamedTuple{ks}, fields) where ks = NamedTuple{ks}(fields)
_reassemble(::Tuple, fields)                   = Tuple(fields)
_leafmeta(proto::TreeNamedTuple) = (;dims = TreeArrays.dims(proto), outer_dim = outerdim(proto))
# a TreeTuple leaf never carries an `outer_dim` (positional fields have no naming axis to
# record); if a future path builds a TreeTuple WITH one, this drops it silently -- widen
# this method (not a bespoke special case) if that ever becomes real.
_leafmeta(proto::TreeTuple)      = (;dims = TreeArrays.dims(proto))
# Reducing an axis of length 0 whose elements are THEMSELVES TreeData: the gather has no
# leaf to push `f` into, and no prototype to rebuild the node's structure from. Unlike a
# zero-length NUMERIC slice (which `f` handles itself -- `nanquantile` of an empty slice is
# NaN, by NaNStatistics' convention), there is no value to invent here without a design
# call. Say so, instead of a bare `BoundsError` from `first` (snag: empty/zero-leaf).
_emptyreduce(sl) = error("TreeArrays: cannot reduce a length-0 axis whose elements are TreeData (got a $(typeof(sl)) of $(length(sl))) -- there is no leaf to push the reduction into. Reduce a dim of the LEAVES instead, and keep the empty axis (it melts to zero rows).")

_leafreduce(f, sl::AbstractArray{<:Union{TreeNamedTuple,TreeTuple}}) = begin
    isempty(sl) && _emptyreduce(sl)
    proto = first(sl)
    ks = keys(parent(proto))
    fields = map(k -> _leafreduce(f, map(el -> parent(el)[k], sl)), ks)
    TreeData(_reassemble(parent(proto), fields), _leafmeta(proto))
end
# `sl` is indexed once per ELEMENT here, never once per element per inner position. The
# distinction is invisible for an `Array` parent (`sl[j]` is a load) but decides the cost of a
# LAZY parent -- `TreeRaggedArray`'s `P<:AbstractArray{<:TreeData}` admits an index-backed array
# that BUILDS each leaf on `getindex`, and the earlier inner-major form (`map(el -> parent(el)[i],
# sl)` under `map(i -> ..., CartesianIndices(p))`) re-walked `sl` for every inner position, so it
# rebuilt each leaf `length(p)` times. A leaf's byte size is proportional to `length(p)`, so that
# made a lazy outer axis pathological in exactly the regime that motivates one (measured:
# n_inner=20_000, n_boot=8 -> 160_001 leaf builds / 4.99 s, vs 8 / 0.003 s here).
# TRADEOFF, deliberate: `ps` pins every leaf's backing array for the duration, so peak memory is
# the whole outer axis. Holding one leaf live at a time instead is only possible when `f` is an
# incremental accumulator; a slice kernel like `quantile` needs the full gathered slice per inner
# position, so its peak is irreducible. Nothing is stacked or pivoted -- `ps` holds the parents'
# own arrays, not a combined copy -- so the no-eager-restructuring invariant is intact.
_leafreduce(f, sl::AbstractArray{<:TreeData}) = begin
    isempty(sl) && _emptyreduce(sl)   # BEFORE `collect`: there is no `proto` to read
    els = collect(sl)   # the one and only pass over `sl`
    proto = first(els)
    p = parent(proto)
    p isa AbstractArray || return _leafreduce(f, map(parent, els))  # scalar leaf: reduce directly, no positions
    _assertconformable(els, size(p))   # `els`, never `sl`: re-walking a lazy outer axis is
    ps = map(parent, els)              # exactly the cost `fcb3f9a` removed
    vals = map(i -> _leafreduce(f, map(P -> P[i], ps)), CartesianIndices(p))
    TreeData(vals, meta(proto))
end
_leafreduce(f, sl::AbstractArray) = f(sl)

# The gather above indexes EVERY leaf with `CartesianIndices` of the FIRST leaf's parent. If
# the leaves have different shapes -- a genuinely ragged nesting whose inner axis was never
# collapsed -- that reads the wrong cells out of the longer leaves and drops their tail
# entirely, returning a plausible-looking wrong answer with no error at all. Reducing a ragged
# OUTER axis is only meaningful once the leaves are conformable (treearrays-use §8: collapse
# the inner ragged axis FIRST, which is what makes the subsequent outer reduce dense). Say so.
function _assertconformable(sl, sz)
    for (j, el) in pairs(sl)
        p = parent(el)
        p isa AbstractArray && size(p) == sz && continue
        got = p isa AbstractArray ? string(size(p)) : "a scalar leaf"
        throw(DimensionMismatch(
            "cannot reduce the outer axis of a ragged tree whose leaves are not conformable: " *
            "leaf 1 has size " * string(sz) * " but leaf " * string(j) * " has " * got * ". " *
            "Collapse the ragged inner axis first (e.g. `mapslices(f, X; dims=:time)`), then " *
            "reduce the outer axis."))
    end
end

# ===================== `dims=` is foundALL, not foundany (decision 1iy1r57) =====================
# `dims=` used to mean "reduce whichever of these names I find here" -- `_splitdims`'s flag is
# literally `foundany`. So a name that resolved NOWHERE reduced nothing and yielded the `missing`
# sentinel: a typo'd dim produced a plausible-looking result rather than an error. A single
# Symbol hid it behind `missing`; a collection made it observable, since the typo's siblings
# reduced normally and the result *looked* reduced. That is precisely what decision 16fwcnx
# forbids -- "under no circumstances silently return a potentially-valid-looking value".
#
# Decidable from the TYPE alone: every dim name is a `TreeDim{N}` type parameter. The walk
# reports `complete = false` at a non-concrete boundary (a JAGGED nesting hides its children's
# names), because absence cannot be PROVEN there -- and rejecting a reduce TreeArrays exists to
# serve would be far worse than missing one typo. So the assert stands down, never guesses.
_alldimnames(::Type{T}) where T<:TreeData = begin
    isconcretetype(T) || return ((), false)          # jagged: cannot see through, cannot prove absence
    own = map(name, Tuple(fieldtype(fieldtype(T, :meta), :dims).parameters))
    cnames, complete = _childdimnames(fieldtype(T, :parent))
    ((own..., cnames...), complete)
end
_childdimnames(::Type{P}) where P<:AbstractArray{<:TreeData}        = _alldimnames(eltype(P))
_childdimnames(::Type{P}) where P<:Tuple{TreeData,Vararg{TreeData}} = _alldimnames(eltype(P))
_childdimnames(::Type{P}) where P<:TreeData                         = _alldimnames(P)   # bookkeeping wrapper
_childdimnames(::Type{P}) where P<:NamedTuple = begin               # a record: union over its fields
    acc, complete = (), true
    for i in 1:fieldcount(P)
        F = fieldtype(P, i)
        nms, ok = F <: TreeData ? _alldimnames(F) : ((), true)
        acc = (acc..., nms...)
        complete &= ok
    end
    (acc, complete)
end
_childdimnames(::Type{P}) where P<:AbstractArray = ((), true)       # dense leaf
_childdimnames(::Type{P}) where P<:Tuple         = ((), true)
_childdimnames(::Type{P}) where P                = ((), true)       # scalar leaf

# The INSTANCE always knows its own dims, even where the type does not. Only walked when the
# type walk came back incomplete AND a name looked absent -- i.e. a jagged tree with a
# suspected typo. Never on the hot path.
_instancedimnames(X::TreeData) = (acc = Symbol[]; _collectdimnames!(acc, X); acc)
function _collectdimnames!(acc, X::TreeData)
    for d in TreeArrays.dims(X); push!(acc, name(d)); end
    _collectchildnames!(acc, parent(X))
end
_collectchildnames!(acc, p::AbstractArray{<:TreeData}) = (for el in p; _collectdimnames!(acc, el); end; acc)
_collectchildnames!(acc, p::Tuple{Vararg{TreeData}})   = (for el in p; _collectdimnames!(acc, el); end; acc)
_collectchildnames!(acc, p::NamedTuple) = (for v in values(p); v isa TreeData && _collectdimnames!(acc, v); end; acc)
_collectchildnames!(acc, p::TreeData)   = _collectdimnames!(acc, p)   # bookkeeping wrapper
_collectchildnames!(acc, p::AbstractArray) = acc                      # dense leaf
_collectchildnames!(acc, p::Tuple)         = acc
_collectchildnames!(acc, p)                = acc                      # scalar leaf

# `@generated`, so the common case folds away at compile time: a correct `dims=` on a concrete
# tree costs nothing at runtime.
@generated function _absentnames(::Type{T}, ::Val{want}) where {T<:TreeData, want}
    names, complete = _alldimnames(T)
    absent = Tuple(filter(nm -> !(nm in names), collect(want)))
    :(($(absent), $(complete), $(Tuple(unique(names)))))
end

# The assert NEVER stands down. An earlier cut skipped the check whenever the type walk was
# incomplete, reasoning that absence could not be *proven* through a jagged boundary. That left
# a silent wrong answer: `dims=(:draw, :drwa)` on a jagged tree reduced `:draw`, dropped the
# typo, and returned a plausible result. Absence is always provable -- just not always from the
# type. Fall back to the instance rather than let a typo through (user, 2026-07-10: "always fail
# loudly instead of doing something unexpected").
@noinline function _assertdimsexist(X::TreeData, ::Val{want}) where want
    absent, complete, have = _absentnames(typeof(X), Val(want))
    isempty(absent) && return nothing
    if !complete                                   # jagged: the type hid some names, the tree has them
        seen = Tuple(unique(_instancedimnames(X)))
        still = Tuple(filter(nm -> !(nm in seen), collect(want)))
        isempty(still) && return nothing
        absent, have = still, seen
    end
    error("TreeArrays: dims=$(want) names " *
          (length(absent) == 1 ? "a dim that exists" : "dims that exist") *
          " nowhere in this tree: $(absent). Available dims: $(have). " *
          "`dims=` is foundALL -- a name that resolves nowhere is a typo, not an empty reduction.")
end

# The PUBLIC entry points assert; the recursion below calls `_mapslices` directly, so a branch
# that legitimately lacks the dim is never mistaken for a typo.
Base.@constprop :aggressive function Base.mapslices(f, X::TreeArray; dims, coords::Bool=false)
    valwant = Val(_dimnames(dims))
    _assertdimsexist(X, valwant)
    _mapslices(_wantcoords(f, Val(coords)), X, valwant)
end
function _mapslices(f, X::TreeArray, valwant::Val{want}) where want
    r = _reduceouter(f, X, valwant)
    isnothing(r) || return r
    alldims = TreeArrays.dims(X)
    # The name exists SOMEWHERE (the public entry proved it) but not on this branch. There is no
    # value to return that is not either a lie or a sentinel, and the sentinel is what 1iy1r57
    # retires: the Tables adapter already declares a heterogeneous-dims shape a non-goal, so a
    # `missing` here could only ever travel to a melt that refuses it. Say so at the source.
    any(nm -> nm in map(name, alldims), want) || error(
        "TreeArrays: cannot reduce dims=$(want) on this branch -- it carries $(map(name, alldims)) " *
        "and none of the requested dims, though they exist elsewhere in the tree. A dim present on " *
        "some branches and absent on others is a heterogeneous-dims shape, which the Tables adapter " *
        "already refuses; reduce a dim the whole branch carries, or split the tree.")
    TreeData(parent(X), (;dims = map(d -> name(d) in want ? sliced(d) : d, alldims)))
end

# Partition a TreeNamedTuple's dims into (inner, ghosts) at compile time: `rec` (the field-
# enumerating axis) is excluded from BOTH -- it's handled separately via `outer_dim`, not
# passed down to children nor carried as a ghost. Same proven shape/reasoning as `_splitdims`
# (a mixed axis/non-axis partition over a heterogeneous tuple) -- `@generated`.
@generated function _splitrecord(alldims::Tuple, ::Val{recname}) where recname
    inner  = Expr[]
    ghosts = Expr[]
    for (i, d) in _dimtypes(alldims)
        if _isaxis(d)
            name(d) == recname || push!(inner, :(alldims[$i]))
        else
            push!(ghosts, :(alldims[$i]))
        end
    end
    :(($(inner...),), ($(ghosts...),))
end

# Drop any ghost already covered by `have` (the sample field's own dims) -- same shape as
# `_splitrecord`/`_splitdims`, `@generated`.
@generated function _exclude(ghosts::Tuple, ::Val{have}) where have
    keep = Expr[]
    for (i, d) in _dimtypes(ghosts)
        name(d) in have || push!(keep, :(ghosts[$i]))
    end
    :(($(keep...),))
end

# `_firstsample` (find the first non-`missing` field) is GONE with 1iy1r57: a field that lacks
# the dim now throws in `_mapslices(::TreeArray)` rather than yielding `missing`, so no field of
# `newfields` can be `missing` and every field is a valid sample.
Base.@constprop :aggressive function Base.mapslices(f, X::TreeNamedTuple; dims, coords::Bool=false)
    valwant = Val(_dimnames(dims))
    _assertdimsexist(X, valwant)
    _mapslices(_wantcoords(f, Val(coords)), X, valwant)
end
function _mapslices(f, X::TreeNamedTuple, valwant::Val{want}) where want
    rec  = outerdim(X)
    inner, ghosts = _splitrecord(TreeArrays.dims(X), Val(name(rec)))   # rec enumerates the fields, not an inner axis
    newfields = map(v -> _mapslices(f, _aschild(v, inner), valwant), parent(X))   # `_mapslices`: children never re-assert
    sample = first(newfields)
    have   = map(name, TreeArrays.dims(sample))
    extra  = _exclude(ghosts, Val(have))
    TreeData(newfields, (;dims = (TreeArrays.dims(sample)..., extra...), outer_dim = rec))
end

Base.@constprop :aggressive function Base.mapslices(f, X::TreeRaggedArray; dims, coords::Bool=false)
    valwant = Val(_dimnames(dims))
    _assertdimsexist(X, valwant)   # a typo, before any structural check can mistake it for a shape
    _mapslices(_wantcoords(f, Val(coords)), X, valwant)
end
function _mapslices(f, X::TreeRaggedArray, valwant::Val{want}) where want
    _isstraddle(X, valwant) && return _pooledstraddle(f, X, valwant)   # outer axis + named leaf-inner dims: pool
    r = _reduceouter(f, X, valwant)
    isnothing(r) || return r
    TreeData(map(el -> _mapslices(f, el, valwant), parent(X)), meta(X))   # `_mapslices`: children never re-assert
end

# A `dims=` that names BOTH this ragged node's outer axis AND a dim living inside its leaves is a
# STRADDLE. The plain gather-loop cannot serve it directly -- it pools along the outer axis at FIXED
# inner positions, so it would reduce the outer axis and leave the named inner one a LIVE axis
# (`foundany`, not `foundall`: a silent under-reduction). But the straddle is a real, well-defined
# operation, not a caller mistake: it is exactly what the DENSE reduce
# `TreeData(arr, :draw,:chain,:param=>…); dims=(:draw,:chain)` already computes -- `eachslice` over the
# KEPT axes hands the kernel the whole pooled (draw x chain) slice per kept index (see the pooled-reduce
# testset). The only reason a ragged nesting couldn't was that it was UNIMPLEMENTED. `_pooledstraddle`
# implements it as a STREAMING pooled gather (peak = ONE kept-index's pooled slice; the per-leaf backing
# arrays are referenced, never copied or stacked), so a consumer holding per-chain @mmap'd matrices as a
# `(chain -> (draw,param))` tree pools across chains WITHOUT an eager `hcat` into a dense block. A
# purely-inner reduce never reaches here (the recursion in `_mapslices` handles it).
_isstraddle(X::TreeRaggedArray, ::Val{want}) where want = begin
    alldims = TreeArrays.dims(X)
    outer   = ntuple(i -> name(alldims[i]), ndims(parent(X)))
    any(nm -> nm in outer, want) || return false          # outer axis not reduced here: not a straddle
    any(nm -> !(nm in map(name, alldims)), want)          # ...and some named dim lives inside the leaves
end

# Pool this ragged node's OUTER axis together with the named LEAF-INNER dims, keeping the un-named inner
# axes. v1 serves the shape a pooled posterior summary needs: a single-level ragged nesting of
# CONFORMABLE, array-backed leaves, with EVERY outer axis reduced. The harder shapes -- a KEPT outer axis,
# record/tuple/doubly-ragged leaves, non-conformable leaves -- each throw BY NAME rather than answer a
# partial or wrong pool (the never-silently-wrong discipline, decisions 16fwcnx / 1iy1r57).
function _pooledstraddle(f, X::TreeRaggedArray, ::Val{want}) where want
    _refusecoords(f, want)          # a pooled bag has no single aligned coordinate vector
    alldims = TreeArrays.dims(X)
    n_outer = ndims(parent(X))
    outer   = ntuple(i -> alldims[i], n_outer)
    all(d -> name(d) in want, outer) || throw(ArgumentError(
        "cannot pool dims " * string(want) * " across a ragged boundary while KEEPING part of this node's " *
        "outer axis " * string(map(name, outer)) * ": keeping one outer axis while pooling another across " *
        "the nesting is not implemented. Reduce every outer axis, or split the tree."))
    isempty(parent(X)) && _emptyreduce(parent(X))
    els   = collect(parent(X))                            # one pass over the outer axis (references, no copy)
    proto = first(els)
    p     = parent(proto)
    p isa AbstractArray || throw(ArgumentError(
        "cannot pool dims " * string(want) * " across a ragged boundary: its leaves are " * string(typeof(proto)) *
        ", not array-backed. Pooling across a record / tuple / doubly-ragged leaf is not implemented -- " *
        "reduce the inner dim(s) first, then the outer axis."))
    _assertconformable(els, size(p))                      # the KEPT inner axes must match across leaves
    keepaxes, keptdims, trailing, _ = _splitdims(TreeArrays.dims(proto), Val(ndims(p)), Val(want))
    ps     = map(parent, els)                             # the leaves' own backing arrays -- never stacked
    ghosts = (trailing..., map(sliced, outer)..., alldims[n_outer+1:end]...)   # reduced inner + outer, sliced
    if isempty(keepaxes)                                  # every inner axis reduced too: one fully-pooled leaf
        _assemble(_leafreduce(f, reduce(vcat, (vec(P) for P in ps))), keptdims, ghosts)
    else                                                  # keep the un-named inner axes; pool per kept index
        sls  = map(P -> _eachslice(P, Val(keepaxes)), ps)
        outs = map(i -> _leafreduce(f, reduce(vcat, (vec(sl[i]) for sl in sls))), CartesianIndices(first(sls)))
        _assemble(outs, keptdims, ghosts)
    end
end

# wrap `outs` (the raw per-slice kernel outputs -- already TreeData/record/scalar pieces,
# never stacked/pivoted) as one TreeData over the kept + reduced-as-ghost dims.
# When ALL of a node's axes were reduced (`keepaxes` empty), `outs` is a SINGLE child TreeData,
# not an array. Wrapping it in a fresh TreeData would build a `TreeData{<:TreeData}` node with no
# use -- an artifact a chained reduction (reduce :subject THEN :draw) then chokes on (`_eltype`
# and `mapslices` have no method for it; Bruno 2026-07-06). Instead MERGE the reduced-as-ghost
# dims straight into the child via the appending constructor (`TreeData(::TreeData, dims...)`,
# types.jl), so the ghosts trail the child's own axes and no wrapper level is created. `keptdims`
# is empty in this branch by construction (a single child <=> `keepaxes` empty), so only
# `trailing` carries dims to merge.
_assemble(outs, keptdims, trailing)           = TreeData(outs, (;dims = (keptdims..., trailing...)))
_assemble(outs::TreeData, keptdims, trailing) = TreeData(outs, keptdims..., trailing...)
