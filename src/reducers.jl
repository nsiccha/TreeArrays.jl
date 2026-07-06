# ===================== reducers =====================
Statistics.mean(X::TreeData; dims=nothing) = isnothing(dims) ? mean(parent(X)) : mapslices(mean, X; dims)
Base.sum(X::TreeData; dims=nothing) = isnothing(dims) ? sum(parent(X)) : mapslices(sum, X; dims)

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
