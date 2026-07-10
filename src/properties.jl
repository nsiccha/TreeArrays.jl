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

Base.@constprop :aggressive function Base.getproperty(X::TreeNamedTuple, s::Symbol)
    P = parent(X)
    hasfield(typeof(P), s) || throw(ArgumentError(
        "TreeNamedTuple has no field `$s`; record axis `$(_recname(X))` has fields $(keys(P))"
    ))
    _aschild(getfield(P, s), _innerdims(X))
end
Base.propertynames(X::TreeNamedTuple) = keys(parent(X))
