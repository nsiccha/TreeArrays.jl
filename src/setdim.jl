unsetdim(X) = X

"""
    setdim(X::TreeData; kwargs...)

!!! danger "Not implemented"
    `setdim` is an exported **stub** and always errors. It is kept as a
    placeholder for re-labelling an axis in place.

    To *restrict* a named axis to a subset of its coordinates, use
    [`selectdim`](@ref), which does exist and copies no data.
"""
setdim(X::TreeData; kwargs...) = error("setdim not implemented")#TreeData(unsetdim(parent(X); kwargs...), (;dims=setdim(meta(X).dims; kwargs...)))
setdim(dims::Tuple; kwargs...) = error("setdim not implemented")#values(merge(), (;kwargs...))

Base.cat(X::TreeData...) = TreeData(X)

# The coordinate values a dim contributes to a sweep. A dim is swept over exactly
# what it *is*: a collection sweeps its elements, and a scalar is a single fixed
# position, so it contributes ZERO array axes (`Iterators.product` treats a Number
# as a 0-dimensional iterable) and lands as a trailing fixed dim -- which is the
# `oni1bc` fixed-vs-axis semantics, arrived at for free. To sweep one value as a
# real length-1 AXIS, pass a 1-tuple: `TreeDim(:schedule, ("s",))`.
#
# An UNLABELLED dim (`values === missing`) is a real axis with no coordinates, so
# there is nothing to sweep. Base's `Iterators.product` reports that as
# `MethodError: no method matching length(::Missing)` from three frames down; say
# what actually went wrong instead.
_sweepvalues(d::TreeDim) = _sweepvalues(name(d), meta(d).values)
_sweepvalues(n::Symbol, ::Missing) = error(
    "map/stack over a TreeDim sweep: dim `$n` is unlabelled (values === missing), so it has " *
    "no coordinates to sweep. Give it a collection, e.g. `TreeDim(:$n, (20, 200))`."
)
_sweepvalues(::Symbol, values) = values

"""
    map(f, d::TreeDim)
    map(f, Iterators.product(dims::TreeDim...))

Build a **scenario sweep**: a [`TreeData`](@ref) whose axes are the swept dims,
in order, and whose every cell is whatever `f` returns.

`f` receives the **coordinate tuple** (the labels, not data), in dim order — so
destructure it:

```julia
ds = (TreeDim(:fit, (:a, :b)), TreeDim(:dose, (20, 200)), TreeDim(:health, ("hi", "lo")))

sweep = map(Iterators.product(ds...)) do (fit, dose, health)
    quantile(draws_for(fit, dose, health), TreeDim(:band, (0.1, 0.5, 0.9)); dims = :draw)
end
# sweep isa TreeData; size(parent(sweep)) == (2, 2, 2); dims are (:fit, :dose, :health)
```

Arity-generic — there is no two-dim limit, and label types may be heterogeneous.
A returned `TreeData` stays a leaf; nothing is densified, and the sweep composes
with the reductions like any other tree.

A **scalar** dim is a fixed position, not an axis: it contributes zero array axes
and lands as a trailing fixed dim. To sweep one value as a real length-1 axis,
pass a 1-tuple — `TreeDim(:schedule, ("s",))`. An **unlabelled** dim has no
coordinates to sweep and errors by name.

!!! note "Splat a `Tuple`, not a generator"
    Splatting a *runtime-length* container of dims is inference-opaque, as any
    runtime-length splat is in Julia. It still works; to also infer, splat a
    `Tuple` (`Tuple(ds)`) or cross a `Tuple`-typed function barrier.
`stack` is accepted as a spelling of `map` here and does the same thing.
"""
Base.map(f, iter::TreeDim) = TreeData(map(f, _sweepvalues(iter)), iter)
Base.stack(f, iter::TreeDim) = map(f, iter)#
Base.stack(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{TreeDim}}}) = map(f, iter)
# N-generic: `Iterators.product` is arity-generic and the dims are forwarded to
# `TreeData` positionally, so the result's dims line up with the parent array's
# axes in order. Verified at arities 1/3/6/8/10 (decision 1qex8lj) -- Bruno's
# `db_profile_plot` sweeps 10.
Base.map(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{TreeDim}}}) = TreeData(
    map(f, Iterators.product(map(_sweepvalues, iter.iterators)...)), iter.iterators...
)
