# ===================== TreeArray (array-backed) array interface =====================
# TreeData is deliberately not an AbstractArray (TreeNamedTuple / TreeRaggedArray aren't
# arrays either); only the array-backed TreeArray alias gets the basic array interface,
# delegating straight through to the backing array. TreeRaggedArray{P<:AbstractArray{<:TreeData}}
# is a strict subconstraint of TreeArray{P<:AbstractArray}, so these methods also apply to
# ragged values at the OUTER level (size/length -> outer count, getindex/iterate -> sub-trees) --
# intentional, sensible outer-level semantics, not a gap.
Base.size(X::TreeArray, args...) = size(parent(X), args...)
Base.length(X::TreeArray) = length(parent(X))
Base.ndims(X::TreeArray) = ndims(parent(X))
Base.eltype(T::Type{<:TreeArray}) = eltype(fieldtype(T, :parent))
Base.axes(X::TreeArray, args...) = axes(parent(X), args...)
Base.getindex(X::TreeArray, i...) = getindex(parent(X), i...)
Base.iterate(X::TreeArray, args...) = iterate(parent(X), args...)
Base.collect(X::TreeArray) = collect(parent(X))
# `Array(X)` extracts the dense backing as a plain `Array` -- the habitual spelling a
# consumer reaches for to feed an external numeric-array API (`MCMCDiagnosticTools.ess`/
# `rhat`, anything typed on `AbstractArray{<:Real,N}`). This does NOT make TreeData an
# array (it stays not-an-AbstractArray): it is a one-way EXTRACTION, exactly like
# `collect`/`parent`. Scoped to the array-backed leaf, it "works if it can" (parent ->
# dense Array) "and fails otherwise" (any other shape has no method -> a MethodError)
# -- user steer on snag treedata-array-o-f8d7994d. Zero-copy stays `parent(X)`; this and
# `collect` mirror each other (a copy), incl. the ragged outer-level semantics above.
Base.Array(X::TreeArray) = Array(parent(X))
