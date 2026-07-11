# ===================== reducers =====================
Statistics.mean(X::TreeData; dims=nothing) = isnothing(dims) ? mean(parent(X)) : mapslices(mean, X; dims)
Base.sum(X::TreeData; dims=nothing) = isnothing(dims) ? sum(parent(X)) : mapslices(sum, X; dims)

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
Statistics.var(X::TreeData; corrected::Bool=true, dims=nothing) =
    isnothing(dims) ? var(parent(X); corrected) : mapslices(v -> var(v; corrected), X; dims)
Statistics.std(X::TreeData; corrected::Bool=true, dims=nothing) =
    isnothing(dims) ? std(parent(X); corrected) : mapslices(v -> std(v; corrected), X; dims)

# quantile delegates to mapslices; pdim bundles the output axis name + values,
# kept EXACTLY as given (Tuple stays Tuple, Vector stays Vector, Number stays
# Number) -- fed RAW to quantile!, which mirrors the same container back into
# the leaf (decision xxmv6c option 2). The tree machinery (_leafreduce) handles
# the resulting Tuple- and scalar-parent leaves directly, including when a
# quantile leaf feeds a CHAINED quantile call. One shared sort buffer.
function Statistics.quantile(X::TreeData, pdim::TreeDim; dims)
    p      = meta(pdim).values
    scratch = _eltype(X)[]
    mapslices(X; dims) do slice
        length(scratch) == length(slice) || resize!(scratch, length(slice))
        copyto!(scratch, slice)
        TreeData(quantile!(scratch, p), pdim)
    end
end

# a NamedTuple `p` packs the SAME one-pass quantile! computation into a WIDE
# record leaf (fields named by `keys(p)`, one value per `values(p)` level)
# instead of a band-axis leaf -- the producer side for the Tables bridge's
# wide-emit. `:quantile` is the record axis's own name -- cosmetic/`show`-only
# post-wide-emit (the field-key column it would have produced in long mode is
# exactly what the wide Tables melt drops).
function Statistics.quantile(X::TreeData, p::NamedTuple; dims)
    scratch = _eltype(X)[]
    mapslices(X; dims) do slice
        length(scratch) == length(slice) || resize!(scratch, length(slice))
        copyto!(scratch, slice)
        TreeData(:quantile => NamedTuple{keys(p)}(quantile!(scratch, values(p))))
    end
end

# A name<->prob spec `:band => (median=0.5, q025=0.025, ...)` reduces to a band AXIS labeled by the
# NAMES (keys, as Symbols) and computed from the PROBS (values). The result is an ordinary
# Symbol-Tuple axis leaf -- no dim-model change, `meta.values` already carries the levels -- so
# `TreeTable(wide=:band)` spreads the levels into columns while `wide=()` keeps them as one long
# `:band` column: orientation stays a VIEW choice (decision e18kfn / option A'), unlike the
# `p::NamedTuple` record method above which bakes WIDE at reduction time. Same one-pass shared-
# scratch quantile! as the TreeDim form; the probs are consumed here, not retained on the axis.
function Statistics.quantile(X::TreeData, (nm, spec)::Pair{Symbol, <:NamedTuple}; dims)
    pdim    = TreeDim(nm, keys(spec))
    probs   = values(spec)
    scratch = _eltype(X)[]
    mapslices(X; dims) do slice
        length(scratch) == length(slice) || resize!(scratch, length(slice))
        copyto!(scratch, slice)
        TreeData(quantile!(scratch, probs), pdim)
    end
end
