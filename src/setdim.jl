unsetdim(X) = X
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

Base.stack(f, iter::TreeDim) = map(f, iter)#
Base.stack(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{TreeDim}}}) = map(f, iter)
Base.map(f, iter::TreeDim) = TreeData(map(f, _sweepvalues(iter)), iter)
# N-generic: `Iterators.product` is arity-generic and the dims are forwarded to
# `TreeData` positionally, so the result's dims line up with the parent array's
# axes in order. Verified at arities 1/3/6/8/10 (decision 1qex8lj) -- Bruno's
# `db_profile_plot` sweeps 10.
Base.map(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{TreeDim}}}) = TreeData(
    map(f, Iterators.product(map(_sweepvalues, iter.iterators)...)), iter.iterators...
)
