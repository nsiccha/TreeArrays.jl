# ===================== record field access: `post.beta` =====================
# A TreeNamedTuple's fields ARE its record axis, so `.` is the natural reader for
# them (user steer, 2026-07-10: "the [BRM] integration could be as easy as adding
# `Base.getproperty`"). Nothing in `src/` dot-accesses a TreeData -- `parent(X)`
# and `meta(X)` go through `getfield` -- so the `.` namespace is free for fields,
# and a record whose field is literally named `parent`/`meta` is unambiguous here.
#
# Descent is EXACTLY what `_mapslices(::TreeNamedTuple)` already does: split off
# the record axis, hand the field its container's inner axes via `_aschild`. So a
# field that is already a TreeData is returned AS IS (zero-copy, it knows its own
# dims) and a raw field is wrapped with the inner axes -- `:draw`/`:chain` survive
# either way, and nothing densifies (lazy assembly, decision 1uzarfr). Ghost dims
# stay on the container, not the child, matching that same descent.

# A TreeNamedTuple built through the `Pair` constructor always carries `outer_dim`;
# one built by handing a NamedTuple straight to `TreeData(x, dims...)` does not.
# `Symbol()` never names a dim, so `_splitrecord` then treats every axis as inner.
# Both branches constant-fold (the meta *type* decides), so this stays type-stable.
_recname(X::TreeNamedTuple) = haskey(meta(X), :outer_dim) ? name(outerdim(X)) : Symbol()
_innerdims(X::TreeNamedTuple) = _splitrecord(TreeArrays.dims(X), Val(_recname(X)))[1]

"""
    X.field

Read a record field without copying it. A field already wrapped in
[`TreeData`](@ref) is returned as stored; a raw field is wrapped with its
container's inner axes. Property access on ragged records visits the leaves.
`propertynames(X)` lists available fields. An unknown field raises an error.
"""
Base.@constprop :aggressive function Base.getproperty(X::TreeNamedTuple, s::Symbol)
    P = parent(X)
    hasfield(typeof(P), s) || throw(ArgumentError(
        "TreeNamedTuple has no field `$s`; record axis `$(_recname(X))` has fields $(keys(P))"
    ))
    _aschild(getfield(P, s), _innerdims(X))
end
Base.propertynames(X::TreeNamedTuple) = keys(parent(X))

# A ragged container of records (todo 1h7ye2g): `X.beta` maps `.beta` over the leaves.
# EAGER outer map, zero-copy children -- the fresh outer array holds one pointer per leaf
# position, not data, so the no-eager-restructuring invariant (which forbids stacking DATA
# into dense blocks) is untouched; a lazy outer view would need a new wrapper type for no
# structural gain. `map` over `(keys, els)` re-wraps at the container's OWN shape -- a
# Matrix of records (the natural shape a chained cross-axis reduce leaves behind) reads to
# a Matrix of children, not a flattened vector wearing the container's two inner dims.
# Leaf keys keep their true position (`CartesianIndex` there), so the foundALL errors below
# name the failing leaf at its actual (row, col). foundALL:
# EVERY leaf must be a record carrying the field -- the first that is not throws BY NAME (leaf
# index + what it carries), never a silent skip. An EMPTY container has no leaf to read from,
# so it throws too (the `_emptyreduce` "no prototype" reasoning). Nested ragged leaves recurse
# through this same method; dense/positional leaves refuse with their own words.
function Base.getproperty(X::TreeRaggedArray, s::Symbol)
    els = parent(X)
    isempty(els) && throw(ArgumentError(
        "cannot read `.$s` from an empty TreeRaggedArray -- there is no leaf to read the field from."))
    TreeData(map((j, el) -> _raggedfield(el, s, j), keys(els), els), meta(X))
end
_raggedfield(el::TreeNamedTuple, s::Symbol, j) =
    hasfield(typeof(parent(el)), s) ? getproperty(el, s) : throw(ArgumentError(
        "cannot read `.$s`: leaf $(_leaflabel(j)) of this TreeRaggedArray is a record without field `$s` " *
        "(it has fields $(keys(parent(el)))). `.` is foundALL -- every leaf must carry the field."))
_raggedfield(el::TreeRaggedArray, s::Symbol, j) = getproperty(el, s)   # nested ragged: recurse
_raggedfield(el::TreeArray, s::Symbol, j) = throw(ArgumentError(
    "cannot read `.$s`: leaf $(_leaflabel(j)) of this TreeRaggedArray is a dense numeric leaf (TreeArray), " *
    "not a record. `.` reads record fields -- index the leaf or split the tree."))
_raggedfield(el::TreeTuple, s::Symbol, j) = throw(ArgumentError(
    "cannot read `.$s`: leaf $(_leaflabel(j)) of this TreeRaggedArray is a positional record (TreeTuple), " *
    "whose fields have no names. `.` reads NAMED record fields -- index the leaf positionally."))
_raggedfield(el::TreeData, s::Symbol, j) = throw(ArgumentError(
    "cannot read `.$s`: leaf $(_leaflabel(j)) of this TreeRaggedArray is a $(typeof(el)), not a named record."))

# Leaf labels carry the leaf's TRUE position: linear for a Vector-backed container,
# `(row, col)` for an N-D one (`keys` of an N-D array is `CartesianIndices` there).
_leaflabel(j::CartesianIndex) = string(Tuple(j))
_leaflabel(j) = string(j)

# The prototype leaf's fields. Deliberately NOT an all-leaves intersection: walking every leaf
# on a listing call would build a lazy outer axis in full (the cost `_leafreduce` avoids), and
# the foundALL access above is the correctness gate -- a heterogeneous field still throws LOUDLY
# when read. An empty/non-record container answers `()` (no accessible fields), total and honest.
function Base.propertynames(X::TreeRaggedArray)
    els = parent(X)
    isempty(els) && return ()
    proto = first(els)
    proto isa TreeNamedTuple || return ()
    keys(parent(proto))
end
