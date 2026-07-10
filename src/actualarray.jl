# ===================== TreeActualArray — opt-in lazy AbstractArray view =====================
# `parent(X)` is an AbstractArray only when the leaf HAPPENS to be array-backed; a
# TreeNamedTuple's parent is a NamedTuple of per-field arrays, so it can't feed an
# `AbstractArray`-typed API (MCMCDiagnosticTools.ess/rhat, any numeric N-D-array function).
# `TreeActualArray(X)` presents a rectangular TreeData as a genuine `AbstractArray{T,N}` — the
# RECORD axis PROMOTED to a real array dimension — lazily and ZERO-COPY (getindex walks the
# tree; nothing is materialized). It is EXPLICITLY opt-in and a SEPARATE type, so `TreeData`
# itself stays not-an-AbstractArray (the §0 dispatch principle is untouched); this is a one-way
# VIEW, exactly like `parent`/`collect`/`Array`. User steer, decision 2026-07-10T18-56-57-801-a967zd.
#
# Axes = X's REAL axes (`_isaxis`) in `dims` order, with a TreeNamedTuple's record axis
# (`outer_dim`, always LAST via the `Pair` constructor) trailing — exactly the (draw, chain,
# param) shape ess/rhat want. Both draw representations arrive at the SAME array: a dense leaf
# `TreeData(arr3d, :draw,:chain,:param=>names)` (real axes ARE parent's axes) and a record
# `TreeData(:param => per-param-matrices, :draw,:chain)` (fields assembled lazily). v1 supports
# an array-backed leaf and a (possibly nested) HOMOGENEOUS NamedTuple record; ragged /
# heterogeneous / Tuple-record shapes error BY NAME (rectangular only — mirrors the Tables
# adapter's contract). Nested records fall out of the recursion.

struct TreeActualArray{T,N,X<:TreeData} <: AbstractArray{T,N}
    tree::X
    size::NTuple{N,Int}
end

function TreeActualArray(X::TreeData)
    sz = _actualsize(X)                      # descends + enforces rectangularity / homogeneity
    TreeActualArray{_eltype(X),length(sz),typeof(X)}(X, sz)
end

Base.size(A::TreeActualArray) = getfield(A, :size)
Base.IndexStyle(::Type{<:TreeActualArray}) = IndexCartesian()
Base.getindex(A::TreeActualArray{T,N}, I::Vararg{Int,N}) where {T,N} = _actualat(getfield(A, :tree), I)::T
# the wrapped tree is reachable, and the dim LABELS survive the crossing into array-land
# (the whole point over a bare `parent(X)`): `parent(A)` is the TreeData, `dims(A)` its axes.
Base.parent(A::TreeActualArray) = getfield(A, :tree)
dims(A::TreeActualArray) = dims(getfield(A, :tree))

# ---- size: descend, promoting the record axis; enforce homogeneity (ragged/heterogeneous error).
_actualsize(X::TreeArray) = size(parent(X))     # numeric leaf: real axes ARE parent's axes
function _actualsize(X::TreeNamedTuple)
    haskey(meta(X), :outer_dim) || error(
        "TreeActualArray: this TreeNamedTuple has no record axis to promote — build it as " *
        "`TreeData(:param => (; fields...), :draw, :chain)` so the fields form an axis")
    p = parent(X)
    isempty(p) && error("TreeActualArray: record `$(name(outerdim(X)))` is empty — no array shape")
    reps = map(_actualsizefield, values(p))
    allequal(reps) || error(
        "TreeActualArray: record `$(name(outerdim(X)))` fields have inconsistent shapes $(reps) — " *
        "a rectangular array needs every field the same shape (feed a homogeneous record or a dense leaf)")
    elts = map(_actualeltfield, values(p))
    allequal(elts) || error(
        "TreeActualArray: record `$(name(outerdim(X)))` fields have differing eltypes $(elts) — not a single-eltype array")
    (first(reps)..., length(p))                 # inner field axes ++ record axis (trailing)
end
_actualsize(X::TreeRaggedArray) = error(
    "TreeActualArray: a ragged / array-of-TreeData tree is not supported yet — feed a dense leaf or a homogeneous NamedTuple record")
_actualsize(X::TreeTuple) = error(
    "TreeActualArray: a Tuple-backed record is not supported yet — use a NamedTuple record (`:param => (; a=…, b=…)`)")

_actualsizefield(v::AbstractArray) = size(v)
_actualsizefield(v::TreeData)      = _actualsize(v)
_actualsizefield(v)                = ()          # scalar field ⇒ contributes no inner axis
_actualeltfield(v::AbstractArray)  = eltype(v)
_actualeltfield(v::TreeData)       = _eltype(v)
_actualeltfield(v)                 = typeof(v)

# ---- the value walk (getindex). I is the full cartesian index in ARRAY-axis order (real axes,
#      dims order, record LAST). A leaf indexes its backing array directly; a record splits off
#      the trailing record index, picks that field (homogeneous ⇒ `values(p)[k]` stays
#      type-stable), and recurses on the remaining inner indices. Nothing is materialized.
_actualat(X::TreeArray, I::NTuple{M,Int}) where M = parent(X)[I...]
function _actualat(X::TreeNamedTuple, I::NTuple{M,Int}) where M
    inner = ntuple(k -> I[k], Val(M - 1))       # compile-time split — M is a type param
    _actualatfield(values(parent(X))[I[M]], inner)
end
_actualat(X::TreeRaggedArray, I::Tuple) = error("TreeActualArray: ragged not supported")   # unreachable (ctor errors) — defensive

_actualatfield(v::AbstractArray, inner::Tuple) = v[inner...]
_actualatfield(v::TreeData, inner::Tuple)      = _actualat(v, inner)
_actualatfield(v, ::Tuple)                     = v            # scalar field terminal
