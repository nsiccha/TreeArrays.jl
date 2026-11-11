# Scenario sweeps: `map` / `stack` over one `TreeDim` or an `Iterators.product` of them
# (treearrays-use §7a). The exported `setdim` stub that used to live here was removed in
# 0.2.0 (decision 1c2lsko): a 1.0 does not export a function that throws on every call, and
# a lazy re-label / fix-a-dim operation can be added later without a breaking change.

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
    map(f, dimension::TreeDim)
    map(f, Iterators.product(dimensions::TreeDim...))

Construct a named scenario grid. A single dimension passes each coordinate to
`f`; a product passes a coordinate tuple in dimension order. Each cell returns
a scalar or a [`TreeData`](@ref). Wrap named tuples as named records. `stack`
accepts the same dimension-sweep forms. Sweep dimensions must have iterable
coordinate collections, rather than unlabelled, fixed, or reduced metadata.
"""
Base.map(f, iter::TreeDim) = TreeData(map(f, _sweepvalues(iter)), iter)
Base.stack(f, iter::TreeDim) = map(f, iter)#
Base.stack(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{TreeDim}}}) = map(f, iter)
# N-generic: `Iterators.product` is arity-generic and the dims are forwarded to
# `TreeData` positionally, so the result's dims line up with the parent array's
# axes in order. Verified at arities 1/3/6/8/10; large
# `db_profile_plot` sweeps 10.
Base.map(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{TreeDim}}}) = TreeData(
    map(f, Iterators.product(map(_sweepvalues, iter.iterators)...)), iter.iterators...
)
