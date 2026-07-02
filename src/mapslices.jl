# ===================== mapslices =====================
# Reduce the named `dims`: apply `f` to each leftover-index slice, keep everything.
# `f` returns a TreeData (or a scalar). Reduced dims stay but become `sliced` (aggregated).
# A requested dim that is absent from a leaf -> `missing` (fixed sentinel).

# Shared "found the axis" bookkeeping for TreeArray/TreeRaggedArray: which parent-array
# positions are being reduced (redaxes) vs kept (keepaxes), and the ghost dims left behind
# for the reduced ones (trailing). Returns `nothing` when `want` doesn't hit a real axis
# here -> the caller decides what that means (a true leaf -> sentinel/idempotent re-slice;
# an intermediate node -> recurse deeper into each element).
function _reduceouter(f, X, want)
    alldims = TreeArrays.dims(X)
    names   = map(name, alldims)
    n_ax    = ndims(parent(X))
    @assert all(_isaxis, alldims[1:n_ax]) "_reduceouter assumes the first $n_ax dims of $(typeof(X)) are exactly the parent array's axes, positionally, in order; got a non-axis dim at position $(findfirst(!_isaxis, alldims[1:n_ax])) (dims = $(names))"
    redaxes = Tuple(i for i in 1:n_ax if names[i] in want)
    isempty(redaxes) && return nothing
    keepaxes = Tuple(i for i in 1:n_ax if !(names[i] in want))
    keptdims = Tuple(alldims[i] for i in keepaxes)
    keptset  = Set(keepaxes)
    trailing = Tuple(name(d) in want ? sliced(d) : d for (i, d) in enumerate(alldims) if !(i in keptset))
    outs = isempty(keepaxes) ? _leafreduce(f, parent(X)) : map(sl -> _leafreduce(f, sl), eachslice(parent(X); dims=keepaxes))
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
_leafreduce(f, sl::AbstractArray{<:TreeData}) = begin
    proto = first(sl)
    p = parent(proto)
    p isa AbstractArray || return _leafreduce(f, map(parent, sl))  # scalar leaf: reduce directly, no positions
    vals = map(i -> _leafreduce(f, map(el -> parent(el)[i], sl)), CartesianIndices(p))
    TreeData(vals, meta(proto))
end
_leafreduce(f, sl::AbstractArray) = f(sl)

function Base.mapslices(f, X::TreeArray; dims)
    want = _dimnames(dims)
    r = _reduceouter(f, X, want)
    isnothing(r) || return r
    alldims = TreeArrays.dims(X)
    any(nm -> nm in map(name, alldims), want) || return missing   # dim absent here -> sentinel
    TreeData(parent(X), (;dims = map(d -> name(d) in want ? sliced(d) : d, alldims)))
end

function Base.mapslices(f, X::TreeNamedTuple; dims)
    want = _dimnames(dims)
    rec  = outerdim(X)
    inner  = Tuple(d for d in TreeArrays.dims(X) if _isaxis(d) && name(d) != name(rec))   # rec enumerates the fields, not an inner axis
    ghosts = Tuple(d for d in TreeArrays.dims(X) if !_isaxis(d))
    newfields = map(v -> mapslices(f, _aschild(v, inner); dims), parent(X))
    any(!ismissing, newfields) || return missing        # no child carried the dim -> sentinel
    sample = first(v for v in newfields if !ismissing(v))
    have   = map(name, TreeArrays.dims(sample))
    extra  = Tuple(g for g in ghosts if !(name(g) in have))
    TreeData(newfields, (;dims = (TreeArrays.dims(sample)..., extra...), outer_dim = rec))
end

function Base.mapslices(f, X::TreeRaggedArray; dims)
    want = _dimnames(dims)
    r = _reduceouter(f, X, want)
    isnothing(r) || return r
    TreeData(map(el -> mapslices(f, el; dims), parent(X)), meta(X))
end

# wrap `outs` (the raw per-slice kernel outputs -- already TreeData/record/scalar pieces,
# never stacked/pivoted) as one TreeData over the kept + reduced-as-ghost dims.
_assemble(outs, keptdims, trailing) = TreeData(outs, (;dims = (keptdims..., trailing...)))
