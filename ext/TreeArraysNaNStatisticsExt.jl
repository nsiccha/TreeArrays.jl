module TreeArraysNaNStatisticsExt

using TreeArrays
using TreeArrays: meta, _eltype, _ScratchKernel
using NaNStatistics: NaNStatistics
using Statistics: quantile!

# NaN-aware quantile on TreeData (todo b1am3w): drop NaNs from each slice
# before quantile!-ing it, so a NaN-containing posterior draw never throws
# (motivated by AoV `lineribbon` raising ArgumentError on NaN). Lives here,
# under NaNStatistics' own name, per user direction -- not a bespoke kwarg on
# Statistics.quantile. p-container mirroring (Tuple/Vector/Number), the reused
# scratch (`_ScratchKernel`: one per call serially, one per task threaded), the
# one-pass-multi-level invariant and the `threads = true` default all match
# TreeArrays' own `Statistics.quantile(X::TreeData, pdim; dims)`.
#
# All-NaN slice returns NaN at every quantile level, matching NaNStatistics'
# own convention (nanquantile(all-NaN, q) === NaN, verified against
# NaNStatistics.jl's `_nanquantile!` source).
function NaNStatistics.nanquantile(X::TreeData, pdim::TreeDim; dims, threads::Bool=true)
    _nanquantile(X, pdim, promote_type(float(_eltype(X)), Float64); dims, threads)
end

# `T` is a real type parameter here (not a captured runtime variable) so the
# all-NaN branch stays concretely inferred -- inlining it back into the
# caller reintroduces per-slice allocation (verified).
function _nanquantile(X::TreeData, pdim::TreeDim, ::Type{T}; dims, threads) where T
    p = meta(pdim).values
    TreeArrays.mapslices(_ScratchKernel(s -> TreeData(_nanquantile!(s, p, T), pdim), _eltype(X)), X; dims, threads)
end

# NamedTuple `p` -- same NaN-compaction, packed into a WIDE record leaf
# (mirrors TreeArrays' own `Statistics.quantile(X, p::NamedTuple; dims)`).
function NaNStatistics.nanquantile(X::TreeData, p::NamedTuple; dims, threads::Bool=true)
    _nanquantile(X, p, promote_type(float(_eltype(X)), Float64); dims, threads)
end

function _nanquantile(X::TreeData, p::NamedTuple, ::Type{T}; dims, threads) where T
    k = _ScratchKernel(s -> TreeData(:quantile => NamedTuple{keys(p)}(_nanquantile!(s, values(p), T))), _eltype(X))
    TreeArrays.mapslices(k, X; dims, threads)
end

# name<->prob spec `:band => (median=0.5, ...)` (mirrors Statistics.quantile's Pair method,
# decision e18kfn/A'): NaN-compacted, emitted as a Symbol-labeled band AXIS (keys=names) computed
# from the probs (values) -- the reduce-to-a-named-axis path for `TreeTable(wide=:band)`.
function NaNStatistics.nanquantile(X::TreeData, (nm, spec)::Pair{Symbol, <:NamedTuple}; dims, threads::Bool=true)
    _nanquantile(X, nm, spec, promote_type(float(_eltype(X)), Float64); dims, threads)
end

function _nanquantile(X::TreeData, nm::Symbol, spec::NamedTuple, ::Type{T}; dims, threads) where T
    pdim  = TreeDim(nm, keys(spec))
    probs = values(spec)
    TreeArrays.mapslices(_ScratchKernel(s -> TreeData(_nanquantile!(s, probs, T), pdim), _eltype(X)), X; dims, threads)
end

# The shared slice body of all three spellings: `s` is the kernel's own scratch, already filled
# with the slice (`_ScratchKernel`), so it may be compacted and shrunk in place -- the next slice
# re-fills it at full length.
function _nanquantile!(s::AbstractVector, p, ::Type{T}) where T
    n = _compactnan!(s)
    resize!(s, n)
    n == 0 ? _nanlike(p, T) : quantile!(s, p)
end

# in-place stable partition: move non-NaN values to the front of `v`, return
# their count. `isnan` is generic on `Real` (always false for non-floats), so
# this is a safe no-op walk for integer-eltype data.
function _compactnan!(v::AbstractVector)
    n = 0
    for i in eachindex(v)
        isnan(v[i]) || (n += 1; v[n] = v[i])
    end
    n
end

_nanlike(p::Number, ::Type{T}) where T = T(NaN)
_nanlike(p::Tuple, ::Type{T}) where T = ntuple(_ -> T(NaN), Val(length(p)))
_nanlike(p::AbstractArray, ::Type{T}) where T = fill(T(NaN), size(p))

end # module
