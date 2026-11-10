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

# ===================== size / axes BY DIM NAME (decision 0jwqiy5) =====================
# `size(X, :draw)` / `axes(X, :draw)`: the dim-name twins of the positional forms, on every tree
# shape. A name this node carries as an AXIS answers from its own backing; a name that lives
# inside the children answers only if every child agrees (a ragged `:time` has no one length, and
# says so); a reduced or fixed dim has no length; an unknown name lists the dims that exist.
# `TreeArray` methods too: `size(::TreeArray, args...)` above would otherwise be ambiguous with
# the `TreeData` one for a Symbol argument.
Base.size(X::TreeData, nm::Symbol)  = _axislength(X, nm)
Base.size(X::TreeArray, nm::Symbol) = _axislength(X, nm)
Base.axes(X::TreeData, nm::Symbol)  = Base.OneTo(_axislength(X, nm))
Base.axes(X::TreeArray, nm::Symbol) = Base.OneTo(_axislength(X, nm))

function _axislength(X::TreeData, nm::Symbol)
    ds = TreeArrays.dims(X)
    i = findfirst(d -> name(d) === nm, ds)
    i === nothing || return _ownaxislength(X, ds[i], i, nm)
    n = _childaxislength(X, nm)
    n === nothing && throw(ArgumentError(
        "TreeArrays: no dim `:$nm` in this tree. Available dims: " * string(Tuple(_instancedimnames(X))) * "."))
    n
end

function _ownaxislength(X::TreeData, d, i, nm)
    _isaxis(d) || throw(ArgumentError("TreeArrays: `:$nm` is " *
        (meta(d).values === nothing ? "a reduced dim" : "a fixed dim (`$(meta(d).values)`)") *
        " here, not an axis, so it has no length."))
    _ownaxislength(X, i, nm)
end
_ownaxislength(X::TreeArray, i, nm) = size(parent(X), i)
_ownaxislength(X::TreeTuple, i, nm) = length(parent(X))
function _ownaxislength(X::TreeNamedTuple, i, nm)
    nm === name(outerdim(X)) && return length(parent(X))
    inner, _ = _splitrecord(TreeArrays.dims(X), Val(name(outerdim(X))))
    _agreed(map(v -> _axislength(_aschild(v, inner), nm), values(parent(X))), nm, "field")
end
_ownaxislength(X::TreeData, i, nm) = throw(ArgumentError(
    "TreeArrays: `:$nm` is an axis of a scalar-backed tree, which has no length to report."))

# Below this node: ask every child; they must agree.
_childaxislength(X::TreeData, nm) = nothing
_childaxislength(X::TreeRaggedArray, nm) = _childrenlength(parent(X), nm)
_childaxislength(X::TreeTuple, nm) = _childrenlength(collect(parent(X)), nm)
function _childaxislength(X::TreeNamedTuple, nm)
    inner, _ = _splitrecord(TreeArrays.dims(X), Val(name(outerdim(X))))
    _childrenlength([_aschild(v, inner) for v in values(parent(X))], nm)
end
function _childrenlength(children, nm)
    isempty(children) && return nothing
    ns = [nm in _instancedimnames(c) ? _axislength(c, nm) : nothing for c in children]
    all(isnothing, ns) && return nothing
    any(isnothing, ns) && throw(ArgumentError(
        "TreeArrays: `:$nm` is a dim of some children of this node but not of others, so it has no one length."))
    _agreed(ns, nm, "child")
end
function _agreed(ns, nm, what)
    n1 = first(ns)
    j = findfirst(!=(n1), ns)
    j === nothing && return n1
    throw(ArgumentError("TreeArrays: `:$nm` differs in length across this node's children ($(what) 1: $n1, " *
        "$(what) $j: $(ns[j])), so it has no one length. Read it per child (`size(parent(X)[j], :$nm)`)."))
end
