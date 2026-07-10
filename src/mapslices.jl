# ===================== mapslices =====================
# Reduce the named `dims`: apply `f` to each leftover-index slice, keep everything.
# `f` returns a TreeData (or a scalar). Reduced dims stay but become `sliced` (aggregated).
# A requested dim that is absent from a leaf -> `missing` (fixed sentinel).

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
    outs = isempty(keepaxes) ? _leafreduce(f, parent(X)) : map(sl -> _leafreduce(f, sl), _eachslice(parent(X), Val(keepaxes)))
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
_leafreduce(f, sl::AbstractArray{<:Union{TreeNamedTuple,TreeTuple}}) = begin
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
    els = collect(sl)   # the one and only pass over `sl`
    proto = first(els)
    p = parent(proto)
    p isa AbstractArray || return _leafreduce(f, map(parent, els))  # scalar leaf: reduce directly, no positions
    ps = map(parent, els)
    vals = map(i -> _leafreduce(f, map(P -> P[i], ps)), CartesianIndices(p))
    TreeData(vals, meta(proto))
end
_leafreduce(f, sl::AbstractArray) = f(sl)

Base.@constprop :aggressive Base.mapslices(f, X::TreeArray; dims) = _mapslices(f, X, Val(_dimnames(dims)))
function _mapslices(f, X::TreeArray, valwant::Val{want}) where want
    r = _reduceouter(f, X, valwant)
    isnothing(r) || return r
    alldims = TreeArrays.dims(X)
    any(nm -> nm in map(name, alldims), want) || return missing   # dim absent here -> sentinel
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

# Find the first non-missing field. Missing-ness is decidable from the TYPE alone (`Missing`
# vs not) -- ordinary multiple dispatch, not a `Val`/`@generated` problem (dev #4.5: dispatch
# instead of a value-level if/elseif).
@inline _firstsample() = missing
@inline _firstsample(v::Missing, rest...) = _firstsample(rest...)
@inline _firstsample(v, rest...) = v

Base.@constprop :aggressive Base.mapslices(f, X::TreeNamedTuple; dims) = _mapslices(f, X, Val(_dimnames(dims)))
function _mapslices(f, X::TreeNamedTuple, valwant::Val{want}) where want
    rec  = outerdim(X)
    inner, ghosts = _splitrecord(TreeArrays.dims(X), Val(name(rec)))   # rec enumerates the fields, not an inner axis
    newfields = map(v -> mapslices(f, _aschild(v, inner); dims=want), parent(X))
    any(!ismissing, newfields) || return missing        # no child carried the dim -> sentinel
    sample = _firstsample(newfields...)
    have   = map(name, TreeArrays.dims(sample))
    extra  = _exclude(ghosts, Val(have))
    TreeData(newfields, (;dims = (TreeArrays.dims(sample)..., extra...), outer_dim = rec))
end

Base.@constprop :aggressive Base.mapslices(f, X::TreeRaggedArray; dims) = _mapslices(f, X, Val(_dimnames(dims)))
function _mapslices(f, X::TreeRaggedArray, valwant::Val{want}) where want
    r = _reduceouter(f, X, valwant)
    isnothing(r) || return r
    TreeData(map(el -> mapslices(f, el; dims=want), parent(X)), meta(X))
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
