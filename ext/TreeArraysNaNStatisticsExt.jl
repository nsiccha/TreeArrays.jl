module TreeArraysNaNStatisticsExt

using TreeArrays
using TreeArrays: meta, _eltype
using NaNStatistics: NaNStatistics
using Statistics: quantile!

# NaN-aware quantile on TreeData (todo b1am3w): drop NaNs from each slice
# before quantile!-ing it, so a NaN-containing posterior draw never throws
# (motivated by AoV `lineribbon` raising ArgumentError on NaN). Lives here,
# under NaNStatistics' own name, per user direction -- not a bespoke kwarg on
# Statistics.quantile. p-container mirroring (Tuple/Vector/Number) and the
# shared-scratch / one-pass-multi-level invariants match TreeArrays' own
# `Statistics.quantile(X::TreeData, pdim; dims)`.
#
# All-NaN slice returns NaN at every quantile level, matching NaNStatistics'
# own convention (nanquantile(all-NaN, q) === NaN, verified against
# NaNStatistics.jl's `_nanquantile!` source).
function NaNStatistics.nanquantile(X::TreeData, pdim::TreeDim; dims)
    _nanquantile(X, pdim, promote_type(float(_eltype(X)), Float64); dims)
end

# `T` is a real type parameter here (not a captured runtime variable) so the
# all-NaN branch stays concretely inferred -- inlining it back into the
# caller reintroduces per-slice allocation (verified).
function _nanquantile(X::TreeData, pdim::TreeDim, ::Type{T}; dims) where T
    p       = meta(pdim).values
    scratch = _eltype(X)[]
    TreeArrays.mapslices(X; dims) do slice
        length(scratch) == length(slice) || resize!(scratch, length(slice))
        copyto!(scratch, slice)
        n = _compactnan!(scratch)
        resize!(scratch, n)
        result = n == 0 ? _nanlike(p, T) : quantile!(scratch, p)
        resize!(scratch, length(slice))
        TreeData(result, pdim)
    end
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
