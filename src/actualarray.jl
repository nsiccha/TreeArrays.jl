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
# an array-backed leaf and a (possibly nested) HOMOGENEOUS NamedTuple record. Numeric
# scalar-terminal grids additionally expose their outer array axes directly. Other sub-trees
# are refused EVEN WHEN rectangular, naming the outer axis; unequal leaves
# additionally name the differing inner axis and lengths (heterogeneous / Tuple-record
# shapes error BY NAME likewise — rectangular only, mirrors the Tables adapter's contract).
# Nested records fall out of the recursion.

"""
    TreeActualArray(X::TreeData)

A lazy, **zero-copy** `AbstractArray{T,N}` view of a rectangular tree — the way
to feed a [`TreeData`](@ref) to a numeric N-D-array API (`ess`, `rhat`, anything
typed on `AbstractArray`) without copying and without losing the dim labels.

A `TreeData` is deliberately not an `AbstractArray`, and `parent(X)` is one only
when the leaf *happens* to be array-backed. `TreeActualArray` is the explicit,
one-way view that crosses the boundary: `getindex` walks the tree, nothing is
materialized, and a [`TreeNamedTuple`](@ref)'s **record axis is promoted to a
real array dimension**.

```julia
A = TreeActualArray(X)   # <: AbstractArray, zero-copy, labels kept, any backing
ess(A); rhat(A)
```

Both draw representations therefore arrive at the *same* array — a dense leaf
`TreeData(arr3d, :draw, :chain, :param => names)` and a record
`TreeData(:param => per_param_matrices, :draw, :chain)`. The axes are `X`'s real
axes in `dims` order, with the record axis trailing — exactly the
`(draw, chain, param)` shape `ess`/`rhat` want.

An N-D outer array of **numeric scalar-terminal** trees also presents as a
numeric array, in the outer array's axis order. This includes a field extracted
from a NamedTuple quantile reduction:

```julia
using NaNStatistics
X = TreeData(reshape(1.0:24.0, 3, 2, 4), :subject, :time => (0.0, 1.0), :draw)
p05 = nanquantile(X, (p05=0.05, p50=0.5); dims=:subject).p05
A = TreeActualArray(p05)           # numeric 2 × 4 view, no dense assembly
A[1, 2]                           # 7.1
coords(dims(A)[1])                 # (0.0, 1.0)
```

Every scalar-grid leaf must hold the same concrete `Number` type and have no
real inner axes; fixed dimensions and reduced ghosts are allowed. Empty grids
must retain a concrete scalar-terminal `TreeData` element type. Outer physical
axes and coordinates are preserved. `parent(A)` recovers the tree and
[`dims`](@ref)`(A)` its dimension metadata, including fixed/ghost dimensions;
those dimensions do not add physical array axes.

The view is read-only and retains its tree for its lifetime. Reads alias the
existing backing; replacing a scalar-grid cell with a compatible scalar tree
is visible through the view. Keep shapes, axes, coordinates and terminal types
unchanged while a view exists. Use `collect(A)` or `Array(A)` for an independent,
mutable numeric array. Construction validates the leaves without packing their
values; a view of a reduced field aliases that field, not the reducer's input.

Other outer arrays of sub-trees are refused even when rectangular — unequal
leaves name the outer axis plus the differing inner axis and lengths — unless
explicit ragged alignment is requested with `dims=` and `align=`. Mixed
scalar/array, nonnumeric, nested-grid and heterogeneous scalar types are
unsupported. A zero-dimensional **array-backed** leaf is not a scalar terminal:
it works as a dense leaf on its own, but a grid of such leaves still takes the
array-subtree refusal path. `Tuple`-record shapes remain unsupported.

Two lower-level ways across the same boundary: `parent(X)` is the backing
verbatim and zero-copy, but a scalar grid's backing contains trees rather than
numbers and a record's backing is a NamedTuple. `collect(X)` / `Array(X)` copy
array-backed storage; `collect(A)` / `Array(A)` materialize the numeric view.
"""
struct TreeActualArray{T,N,X<:TreeData} <: AbstractArray{T,N}
    tree::X
    size::NTuple{N,Int}
end

TreeActualArray(X::TreeData) = _actualarray(X)
function _actualarray(X::TreeData)
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
# An outer axis of sub-trees is refused even when the leaves agree — but UNEQUAL
# leaves get their exact words: which inner axis differs, and how (snag
# treeactualarray-998ba536: unequal per-chain `:draw` counts must error BY NAME).
_actualleafsig(l::TreeData) = (name.(dims(l)), _actualsize(l))
function _actualsize(X::TreeRaggedArray)
    outer = dims(X)
    who = isempty(outer) ? "an outer array-of-trees" : "outer axis `:$(name(first(outer)))`"
    leaves = parent(X)
    n = length(leaves)
    sigs = map(_actualleafsig, leaves)
    allequal(sigs) && error(
        "TreeActualArray: $who holds $n identically-shaped sub-trees — an array-of-trees " *
        "is not a supported shape yet, even when rectangular (slicing to a common length does " *
        "not unblock it) — feed a dense leaf (`TreeData(arr3d, :draw, :chain, :param)`) or a " *
        "homogeneous NamedTuple record (`TreeData(:param => (; …), :draw, :chain)`)")
    (names1, sizes1) = sigs[1]
    k = findfirst(s -> s != sigs[1], sigs)
    (namesk, sizesk) = sigs[k]
    detail = if namesk != names1
        "leaf $k carries axes $namesk where leaf 1 carries $names1"
    else
        j = findfirst(i -> sizes1[i] != sizesk[i], 1:min(length(sizes1), length(sizesk)))
        j === nothing ? "leaf $k has rank $(length(sizesk)) where leaf 1 has rank $(length(sizes1))" :
            "axis `:$(names1[j])` lengths differ $(map(s -> s[2][j], sigs))"
    end
    error("TreeActualArray: $who holds $n sub-trees with differing shapes — $detail — " *
        "a rectangular array needs every leaf the same shape (feed a dense leaf or a homogeneous NamedTuple record)")
end
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

# A scalar-terminal grid already has its complete physical shape in the outer
# array. Keep that array (and its axes) intact; read just the selected terminal.
# This is a separate view, like the aligned-ragged view below, so the established
# dense/record value walk and axis order stay unchanged.
struct _ScalarGridTreeActualArray{T,N,X,A} <: AbstractArray{T,N}
    tree::X
    shape::NTuple{N,Int}
    outer_axes::A
end

Base.size(A::_ScalarGridTreeActualArray) = getfield(A, :shape)
Base.axes(A::_ScalarGridTreeActualArray) = getfield(A, :outer_axes)
Base.IndexStyle(::Type{<:_ScalarGridTreeActualArray}) = IndexCartesian()
Base.parent(A::_ScalarGridTreeActualArray) = getfield(A, :tree)
dims(A::_ScalarGridTreeActualArray) = dims(parent(A))
@inline function Base.getindex(A::_ScalarGridTreeActualArray{T,N},
        I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    parent(parent(parent(A))[I...])::T
end

# Scalar eligibility is a representation check, never the ((), ()) shape
# signature: a zero-dimensional AbstractArray is still an array-backed leaf.
_actualscalarterminal(::TreeData) = true
_actualscalarterminal(::Union{TreeArray,TreeNamedTuple,TreeTuple}) = false
_actualscalarterminal(::TreeRaggedArray) = true  # validate/refuse nested grids by name
_actualscalarterminaltype(::Type) = false
_actualscalarterminaltype(::Type{<:TreeData{<:Number}}) = true

function _actualscalargridtype(::Type{TreeData{P,M}}) where {P<:Number,M}
    isconcretetype(TreeData{P,M}) || error(
        "TreeActualArray scalar grid: an empty grid must retain a concrete numeric scalar TreeData element type")
    isconcretetype(P) || error("TreeActualArray scalar grid: the numeric scalar type must be concrete")
    ds = fieldtype(M, :dims)
    any(_isaxis, fieldtypes(ds)) && error(
        "TreeActualArray scalar grid: scalar terminals cannot carry real inner axes -- " *
        "only fixed dimensions and reduced ghosts are supported on a scalar leaf")
    P
end
_actualscalargridtype(::Type{L}) where L = error(
    "TreeActualArray scalar grid: every leaf must hold a numeric scalar terminal; got $L -- " *
    "mixed scalar/array leaves, nonnumeric terminals and nested containers are unsupported " *
    "(a zero-dimensional array-backed leaf is an array, not a scalar terminal). " *
    "An empty grid must retain a concrete numeric scalar TreeData element type")

function _actualscalargrid(tree::TreeRaggedArray)
    leaves = parent(tree)
    T = _actualscalargridtype(isempty(leaves) ? eltype(leaves) : typeof(first(leaves)))
    for (j, leaf) in pairs(leaves)
        S = _actualscalargridtype(typeof(leaf))
        S === T || error(
            "TreeActualArray scalar grid: every leaf must have the same numeric scalar type; " *
            "leaf $j holds $S where the first holds $T -- no implicit promotion")
    end
    axis_dims = filter(_isaxis, dims(tree))
    length(axis_dims) == ndims(leaves) || error(
        "TreeActualArray scalar grid: every outer array axis must have one dimension label")
    for (j, d) in enumerate(axis_dims)
        labels = meta(d).values
        (labels === missing || length(labels) == size(leaves, j)) || error(
            "TreeActualArray scalar grid: outer axis `:$(name(d))` has $(length(labels)) " *
            "coordinates for $(size(leaves, j)) positions")
    end
    _ScalarGridTreeActualArray{T,ndims(leaves),typeof(tree),typeof(axes(leaves))}(
        tree, size(leaves), axes(leaves))
end

function _actualarray(tree::TreeRaggedArray)
    leaves = parent(tree)
    # Empty grids have no sample: only their retained element type can prove
    # scalar eligibility. Non-empty grids validate every leaf, including broad
    # containers whose actual terminals nevertheless share one numeric type.
    scalar = isempty(leaves) ? _actualscalarterminaltype(eltype(leaves)) :
        any(_actualscalarterminal, leaves)
    scalar && return _actualscalargrid(tree)
    _actualsize(tree)                         # existing array/record-subtree refusal
end

# ===================== ragged alignment views (todo 1l53kom) =====================
# Spliced upstream from Bruno's `web-pkpd/src/treearrays_compat.jl` bridge (nearly verbatim --
# same struct shape, same `dims=`/`align=` keywords, same three policies), plus the two pieces
# the bridge deferred to upstream: a coordinate policy for a LABELLED ragged axis, and
# missing-safe / structure-loud sibling checks.
#
# A one-level ragged tree whose leaves differ ONLY along one named inner axis presents as a
# lazy, zero-copy `AbstractArray`: the outer sibling axis is inserted immediately AFTER the
# aligned inner axis, so `:chain -> (:draw x :param)` becomes `:draw x :chain x :param` --
# exactly the `(iterations, chains, parameters)` order `ess`/`rhat` want -- and
# `:subject -> :draw` becomes `:draw x :subject`. Values are read through their existing
# leaves (only per-leaf INDEX vectors are stored, never a dense cube); `dims(A)` keeps the
# labels, with the aligned axis carrying the coordinates actually read (see below).
"""
    TreeActualArray(X::TreeRaggedArray; dims, align)

A lazy, zero-copy `AbstractArray` view of a one-level ragged tree whose leaves
differ only along the inner axis `dims`, aligned under the named policy. The
caller must name **both** — there is no implicit ragged-to-rectangular
conversion, and a half-named call throws for the missing keyword. A bare
`TreeActualArray(X)` on array-backed sub-trees keeps refusing; homogeneous
numeric scalar-terminal grids instead expose their outer axes without alignment.

```julia
A = TreeActualArray(chains; dims = :draw, align = :truncate_min)
ess(A); rhat(A)   # any `AbstractArray`-typed consumer, no overloads needed
```

Policies:

  - `:truncate_min` — head truncation: every leaf contributes its first `target`
    positions, where `target` is the minimum sibling length.
  - `:thin_min` — equidistant thinning: every leaf contributes `target` evenly
    spread positions (`round.(Int, range(1, n; length = target))`).
  - `:error_equal` — refuse unless every sibling already has the same length.

The aligned axis keeps its coordinates under one uniform rule: the READ
positions' coordinates must agree across leaves (compared content-wise, so a
`Tuple` beside a `Vector` still aligns), and the view carries the prototype's.
Truncation reads positions `1:target` (heads must agree — a chain continued
from iteration 101 is not the same iterations 1–60); thinning reads evenly
spread positions per leaf (which agree only when the leaves share a grid, so a
labelled unevenly-sized thin refuses rather than label positions that mean
different things per leaf). Unlabelled axes align vacuously. Non-aligned leaf
axes must match in length and coordinates, every leaf must be array-backed at
the same rank with the same dim structure and eltype, and the outer container
must be a one-dimensional `Vector` of siblings — anything else throws by name.
"""
struct _AlignedRaggedTreeActualArray{T,N,X,I,D} <: AbstractArray{T,N}
    tree::X
    aligned_indices::I
    aligned_pos::Int
    outer_pos::Int
    shape::NTuple{N,Int}
    tree_dims::D
end

Base.size(A::_AlignedRaggedTreeActualArray) = getfield(A, :shape)
Base.IndexStyle(::Type{<:_AlignedRaggedTreeActualArray}) = IndexCartesian()
Base.parent(A::_AlignedRaggedTreeActualArray) = getfield(A, :tree)
dims(A::_AlignedRaggedTreeActualArray) = getfield(A, :tree_dims)

@inline function Base.getindex(A::_AlignedRaggedTreeActualArray{T,N},
        I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    outer_pos = getfield(A, :outer_pos)
    aligned_pos = getfield(A, :aligned_pos)
    sibling = I[outer_pos]
    leaf = parent(getfield(A, :tree))[sibling]
    child_I = ntuple(Val(N - 1)) do child_pos
        out_pos = child_pos < outer_pos ? child_pos : child_pos + 1
        child_pos == aligned_pos ?
            getfield(A, :aligned_indices)[sibling][I[out_pos]] :
            I[out_pos]
    end
    parent(leaf)[child_I...]
end

function TreeActualArray(tree::TreeRaggedArray; dims::Union{Symbol,Nothing}=nothing,
        align::Union{Symbol,Nothing}=nothing)
    # This method is positionally MORE specific than the bare `TreeActualArray(X)` constructor,
    # so a bare ragged call lands here too -- scalar grids use their outer axes,
    # while other subtrees retain the original refusal. A HALF-named
    # call is not a bare call: each missing keyword throws for itself, never a default policy.
    if isnothing(dims) && isnothing(align)
        return _actualarray(tree)
    end
    isnothing(dims) && error(
        "TreeActualArray ragged alignment: `dims=` naming the ragged inner axis is required -- " *
        "there is no implicit ragged-to-rectangular conversion")
    isnothing(align) && error(
        "TreeActualArray ragged alignment: `align=` naming the policy " *
        "(:truncate_min, :thin_min, or :error_equal) is required -- there is no default policy")
    leaves = parent(tree)
    leaves isa AbstractVector || error(
        "TreeActualArray ragged alignment: expected a one-dimensional outer sibling axis")
    isempty(leaves) && error("TreeActualArray ragged alignment: no siblings")
    outer_dims = TreeArrays.dims(tree)
    length(outer_dims) == 1 || error(
        "TreeActualArray ragged alignment: expected one outer sibling axis")

    first_leaf = first(leaves)
    all(leaf -> parent(leaf) isa AbstractArray, leaves) || error(
        "TreeActualArray ragged alignment: every leaf must be array-backed")
    child_dims = TreeArrays.dims(first_leaf)
    child_ndims = ndims(parent(first_leaf))
    length(child_dims) >= child_ndims || error(
        "TreeActualArray ragged alignment: every physical leaf axis must be named")
    # Physical axes come first; fixed/aggregated ghost dims may trail after a reduction.
    # Only physical axes cross into AbstractArray land.
    axis_dims = child_dims[1:child_ndims]
    aligned_pos = findfirst(d -> TreeArrays.name(d) === dims, axis_dims)
    isnothing(aligned_pos) && error(
        "TreeActualArray ragged alignment: no inner axis named `:$dims`")
    _dimkind(axis_dims[aligned_pos]) === :axis || error(
        "TreeActualArray ragged alignment: `:$dims` is a $(_dimkind(axis_dims[aligned_pos])) dim, " *
        "not a real axis -- only a real axis has positions to align")

    # Every sibling must carry the same physical-axis structure at the same rank -- a leaf
    # missing an axis, or carrying a pre-reduced ghost where the prototype has a live axis
    # (mixed provenance), would silently misalign positions. Fixed-dim VALUES are not
    # cross-checked (decision 4b3vcd's beta case, same as the pooled straddle).
    refsig = map(d -> TreeArrays.name(d) => _dimkind(d), child_dims)
    for (j, leaf) in pairs(leaves)
        j == 1 && continue
        ok = ndims(parent(leaf)) == child_ndims &&
            map(d -> TreeArrays.name(d) => _dimkind(d), TreeArrays.dims(leaf)) == refsig
        ok || error("TreeActualArray ragged alignment: leaf $j carries a different dim " *
                    "structure than leaf 1 -- align leaves that share one axis layout")
    end

    T = eltype(parent(first_leaf))
    all(leaf -> eltype(parent(leaf)) === T, leaves) || error(
        "TreeActualArray ragged alignment: every leaf must carry the same numeric eltype")
    for pos in eachindex(axis_dims)
        pos == aligned_pos && continue
        n = size(parent(first_leaf), pos)
        all(leaf -> size(parent(leaf), pos) == n, leaves) || error(
            "TreeActualArray ragged alignment: non-aligned axis `:$(TreeArrays.name(axis_dims[pos]))` differs across leaves")
        # Missing-safe (an unlabelled axis is `missing`, and `missing == missing` is `missing`,
        # not `true`): two unlabelled axes agree, a labelled/unlabelled pair does not.
        values = TreeArrays.meta(axis_dims[pos]).values
        all(leaf -> _poolcoordsagree(TreeArrays.meta(TreeArrays.dims(leaf)[pos]).values, values), leaves) || error(
            "TreeActualArray ragged alignment: non-aligned axis `:$(TreeArrays.name(axis_dims[pos]))` has inconsistent coordinates")
    end

    lengths = [size(parent(leaf), aligned_pos) for leaf in leaves]
    target = minimum(lengths)
    target > 0 || error("TreeActualArray ragged alignment: leaves contain no aligned positions")
    aligned_indices = if align === :error_equal
        allequal(lengths) || error(
            "TreeActualArray ragged alignment: align=:error_equal requires equal lengths; got $lengths")
        [Base.OneTo(target) for _ in leaves]
    elseif align === :truncate_min
        [Base.OneTo(target) for _ in leaves]
    elseif align === :thin_min
        [round.(Int, range(1, n; length=target)) for n in lengths]
    else
        error("TreeActualArray ragged alignment: unknown align=$align; " *
              "expected :error_equal, :truncate_min, or :thin_min")
    end

    # The coordinate policy: the READ positions' coordinates must agree across leaves (an
    # unlabelled axis agrees vacuously); the view carries the prototype's read coordinates.
    # Truncation preserves the label container (`values[1:target]`); thinning re-samples to a
    # `Vector`, since evenly-spread positions are not a range/tuple slice of the original.
    protovals = TreeArrays.meta(axis_dims[aligned_pos]).values
    aligned_values = if protovals === missing
        missing
    else
        reads = [TreeArrays.meta(TreeArrays.dims(leaf)[aligned_pos]).values for leaf in leaves]
        # A leaf that is UNLABELLED where the prototype is labelled reads `missing` -- which
        # disagrees with every label vector below (loud, by name), never silently.
        readcoords(v, idx) = v === missing ? missing : [v[i] for i in idx]
        ref = readcoords(reads[1], aligned_indices[1])
        for k in 2:length(leaves)
            _poolcoordsagree(readcoords(reads[k], aligned_indices[k]), ref) || error(
                "TreeActualArray ragged alignment: aligned axis `:$dims` reads different coordinates " *
                "per leaf under align=$align (leaf $k disagrees with leaf 1) -- no shared labels exist " *
                "for the aligned positions")
        end
        align === :thin_min ? ref : protovals[1:target]
    end

    outer_pos = aligned_pos + 1
    child_shape = ntuple(pos -> pos == aligned_pos ? target :
        size(parent(first_leaf), pos), child_ndims)
    shape = (child_shape[1:aligned_pos]..., length(leaves),
        child_shape[aligned_pos+1:end]...)
    aligned_dim = TreeDim(TreeArrays.name(axis_dims[aligned_pos]), aligned_values)
    out_dims = (axis_dims[1:aligned_pos-1]..., aligned_dim, first(outer_dims),
        axis_dims[aligned_pos+1:end]...)
    N = child_ndims + 1
    _AlignedRaggedTreeActualArray{T,N,typeof(tree),typeof(aligned_indices),typeof(out_dims)}(
        tree, aligned_indices, aligned_pos, outer_pos, shape, out_dims)
end
