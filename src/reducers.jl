# ===================== reducers =====================
"""
    mean(X::TreeData; dims=nothing, threads=true, align=nothing)

Mean over named dimensions, using [`mapslices`](@ref). With `dims=nothing`,
delegate to `mean(parent(X))`; this is a scalar mean for numeric array backing,
not a recursive mean over arbitrary records. `threads=false` disables parallel
slice evaluation. `align` selects a ragged coordinate policy.
"""
Statistics.mean(X::TreeData; dims=nothing, threads::Bool=true, align=nothing) =
    isnothing(dims) ? mean(parent(X)) : mapslices(mean, X; dims, threads, align)

"""
    sum(X::TreeData; dims=nothing, threads=true, align=nothing)

Sum over named dimensions, using [`mapslices`](@ref). With `dims=nothing`,
delegate to `sum(parent(X))`. Threading and alignment follow [`mean`](@ref).
"""
Base.sum(X::TreeData; dims=nothing, threads::Bool=true, align=nothing) =
    isnothing(dims) ? sum(parent(X)) : mapslices(sum, X; dims, threads, align)

# std / var mirror mean/sum: delegate to mapslices, so they compose with the whole reduction
# machinery -- a purely-inner reduce, a dense multi-axis pool, AND the ragged (:draw,:chain)
# straddle (`_pooledstraddle`, mapslices.jl) all hand the kernel the WHOLE pooled slice, one bag.
# So `std(coll; dims=(:draw,:chain))` on a `(chain -> (draw,param))` ragged tree is byte-identical
# to the dense `std(TreeData(arr3d,:draw,:chain,:param); dims=(:draw,:chain))` -- a pooled variance
# is one pass over the whole (draw x chain) bag, NOT the composition of two reductions.
# `corrected` is forwarded: the default `true` gives the SAMPLE variance (÷(n-1)) -- Statistics'
# own default, matching `std(some_vector)` -- and `corrected=false` gives the population (÷n) form.
# `mean=` is intentionally NOT forwarded: one precomputed mean cannot apply across the per-slice
# reductions. The closure `v -> var(v; corrected)` still specializes `mapslices` on its concrete
# type (§5) -- it forwards a Bool, it does not dispatch per element.
"""
    var(X::TreeData; dims=nothing, corrected=true, threads=true, align=nothing)

Variance over named dimensions. `corrected=true` uses the sample denominator;
`false` uses the population denominator. Pool multiple sample dimensions in one
call rather than composing variances. A precomputed `mean` keyword is not
supported. With `dims=nothing`, delegate to `var(parent(X); corrected)`.
Threading and alignment follow [`mean`](@ref).
"""
Statistics.var(X::TreeData; corrected::Bool=true, dims=nothing, threads::Bool=true, align=nothing) =
    isnothing(dims) ? var(parent(X); corrected) : mapslices(v -> var(v; corrected), X; dims, threads, align)

"""
    std(X::TreeData; dims=nothing, corrected=true, threads=true, align=nothing)

Standard deviation over named dimensions, with the options of [`var`](@ref).
With `dims=nothing`, delegate to `std(parent(X); corrected)`.
"""
Statistics.std(X::TreeData; corrected::Bool=true, dims=nothing, threads::Bool=true, align=nothing) =
    isnothing(dims) ? std(parent(X); corrected) : mapslices(v -> std(v; corrected), X; dims, threads, align)

# quantile delegates to mapslices; pdim bundles the output axis name + values,
# kept EXACTLY as given (Tuple stays Tuple, Vector stays Vector, Number stays
# Number) -- fed RAW to quantile!, which mirrors the same container back into
# the leaf (decision xxmv6c option 2). The tree machinery (_leafreduce) handles
# the resulting Tuple- and scalar-parent leaves directly, including when a
# quantile leaf feeds a CHAINED quantile call. One sort buffer per task (`_ScratchKernel`,
# threading.jl): one per call on the serial path.
"""
    quantile(X::TreeData, probabilities::TreeDim; dims, threads=true, align=nothing)
    quantile(X::TreeData, probabilities::NamedTuple; dims, threads=true, align=nothing)
    quantile(X::TreeData, name => probabilities::NamedTuple; dims, threads=true, align=nothing)

Compute the requested quantile levels together for each slice. A `TreeDim`
creates a probability axis; its coordinate container is preserved. A scalar
probability in a `TreeDim` creates fixed metadata rather than a varying axis.
A named tuple creates a record. A name paired with a named tuple creates an axis
labelled by the tuple's keys, with probabilities taken from its values.

Each slice is copied into reusable scratch storage before sorting; inputs are
not sorted in place. `NaN` raises an error. Load NaNStatistics for `nanquantile`
with the same forms; it drops NaNs per slice and returns NaN for an empty valid
slice. Threading and ragged alignment follow [`mapslices`](@ref), with threading
enabled by default here.
"""
function Statistics.quantile(X::TreeData, pdim::TreeDim; dims, threads::Bool=true, align=nothing)
    p = meta(pdim).values
    mapslices(_ScratchKernel(s -> TreeData(quantile!(s, p), pdim), _eltype(X)), X; dims, threads, align)
end

# a NamedTuple `p` packs the SAME one-pass quantile! computation into a WIDE
# record leaf (fields named by `keys(p)`, one value per `values(p)` level)
# instead of a band-axis leaf -- the producer side for the Tables bridge's
# wide-emit. `:quantile` is the record axis's own name -- cosmetic/`show`-only
# post-wide-emit (the field-key column it would have produced in long mode is
# exactly what the wide Tables melt drops).
function Statistics.quantile(X::TreeData, p::NamedTuple; dims, threads::Bool=true, align=nothing)
    k = _ScratchKernel(s -> TreeData(:quantile => NamedTuple{keys(p)}(quantile!(s, values(p)))), _eltype(X))
    mapslices(k, X; dims, threads, align)
end

# A name<->prob spec `:band => (median=0.5, q025=0.025, ...)` reduces to a band AXIS labeled by the
# NAMES (keys, as Symbols) and computed from the PROBS (values). The result is an ordinary
# Symbol-Tuple axis leaf -- no dim-model change, `meta.values` already carries the levels -- so
# `TreeTable(wide=:band)` spreads the levels into columns while `wide=()` keeps them as one long
# `:band` column: orientation stays a VIEW choice (decision e18kfn / option A'), unlike the
# `p::NamedTuple` record method above which bakes WIDE at reduction time. Same one-pass
# reused-scratch quantile! as the TreeDim form; the probs are consumed here, not retained on the axis.
function Statistics.quantile(X::TreeData, (nm, spec)::Pair{Symbol, <:NamedTuple}; dims, threads::Bool=true, align=nothing)
    pdim  = TreeDim(nm, keys(spec))
    probs = values(spec)
    mapslices(_ScratchKernel(s -> TreeData(quantile!(s, probs), pdim), _eltype(X)), X; dims, threads, align)
end

# Highest-density intervals (the brms/bambi-style summary): the SHORTEST interval holding at
# least `mass` of the draws. Computed on a sorted copy -- never mutating the caller's slice --
# by scanning the fixed-count windows, so ties resolve deterministically to the FIRST (lowest)
# interval. An HDI is a data interval, not an interpolation: its endpoints are always draws.
_checkhdimass(mass::Real) =
    0 < mass <= 1 || throw(ArgumentError("hdi mass must lie in (0, 1] (got $mass)."))
_checkhdimass(mass) =
    throw(ArgumentError("hdi mass must be a Real in (0, 1] (got $mass)."))

function _hdi_sorted!(s::AbstractVector, mass::Real)
    n = length(s)
    n == 0 && throw(ArgumentError(
        "hdi of an empty slice is undefined (mass = $mass)."))
    any(isnan, s) && throw(ArgumentError(
        "hdi throws on NaN, matching quantile!: drop NaNs first (`filter(!isnan, y)`)."))
    k = ceil(Int, mass * n)                       # draws the interval must cover (1 <= k <= n)
    best, bestw = 1, s[k] - s[1]
    for i in 2:n-k+1
        w = s[i+k-1] - s[i]
        w < bestw && (best = i; bestw = w)        # strict `<`: ties keep the FIRST window
    end
    (; lower=s[best], upper=s[best+k-1])
end

"""
    hdi(y::AbstractVector, mass=0.9)
    hdi(X::TreeData; dims, mass=0.9, threads=true, align=nothing)

Return the shortest sample interval containing at least `mass` of the draws,
as `(lower, upper)` named fields. Endpoints are observed draws; tied widths
choose the lowest interval. `mass` must be in `(0, 1]`. Empty or NaN-containing
samples raise errors. A single interval can span a gap in multimodal samples.

The tree method returns a record per slice through [`mapslices`](@ref). It
uses scratch storage for sorting and does not mutate the input.
"""
hdi(y::AbstractVector, mass=0.9) = (_checkhdimass(mass); _hdi_sorted!(sort(y), mass))
function hdi(X::TreeData; dims, mass=0.9, threads::Bool=true, align=nothing)
    _checkhdimass(mass)
    k = _ScratchKernel(_eltype(X)) do s
        sort!(s)
        (; lower, upper) = _hdi_sorted!(s, mass)
        TreeData(:hdi => (; lower, upper))
    end
    mapslices(k, X; dims, threads, align)
end
