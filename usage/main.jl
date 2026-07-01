using TreeArrays, DataFrames, LazyArrays, Statistics


struct TreeDim{N,M<:NamedTuple}
    meta::M
    TreeDim(N, meta::NamedTuple) = new{N,typeof(meta)}(meta)
end
TreeDim(N, values) = TreeDim(N, (;values))
TreeDim(N::Symbol) = TreeDim(N, missing)
TreeDim((N,values)::Pair) = TreeDim(N, values)
name(::TreeDim{N}) where N = N
name(::Type{<:TreeDim{N}}) where N = N
meta(X::TreeDim) = getfield(X, :meta)
sliced(X::TreeDim) = TreeDim(name(X), (;values=nothing))
# keep-as-provided: a collection value is a real axis (iterate its coords); a scalar,
# `missing` (unlabelled) or `nothing` (aggregated) value is a single fixed position.
_coords(v::Union{Tuple,AbstractArray,AbstractRange}) = v
_coords(v) = (v,)
Base.length(X::TreeDim) = length(_coords(meta(X).values))
Base.iterate(X::TreeDim, args...) = iterate(_coords(meta(X).values), args...)
struct TreeData{P,M<:NamedTuple}
    parent::P
    meta::M
    TreeData(parent, meta::NamedTuple) = new{typeof(parent), typeof(meta)}(parent, meta)
end
Base.parent(X::TreeData) = getfield(X, :parent)
meta(X::TreeData) = getfield(X, :meta)
TreeData(dims::Symbol...) = (x, vals...)->TreeData(x, map(TreeDim, dims, vals)...)
TreeData(x, dims...; kwargs...) = TreeData(x, map(TreeDim, dims)..., map(TreeDim, keys(kwargs), values(kwargs))...)
TreeData(x, dims::TreeDim...) = TreeData(x, (;dims))
# a TreeNamedTuple keeps its record axis in the type as a SEPARATE `outer_dim`: being the
# record axis is a property of the container (which axis enumerates the fields), not of a dim.
TreeData((name, X)::Pair{Symbol,<:NamedTuple}, dims::TreeDim...) = TreeData(
    X, (;dims, outer_dim = TreeDim(name, keys(X)))
)
TreeData(X::TreeData, dims::TreeDim...) = TreeData(parent(X), meta(X).dims..., dims...)
TreeNamedTuple{P<:NamedTuple,M<:NamedTuple} = TreeData{P,M}
TreeRaggedArray{P<:AbstractArray{<:TreeData},M<:NamedTuple} = TreeData{P,M}
TreeArray{P<:AbstractArray,M<:NamedTuple} = TreeData{P,M}
# convenience accessors (avoid spelling out `meta(...).field` everywhere)
dims(X::TreeData) = meta(X).dims                 # the tree's (inner) axes
outerdim(X::TreeData) = meta(X).outer_dim        # a TreeNamedTuple's record axis

# leaf numeric eltype: recurse through NamedTuple / ragged nesting down to the backing array.
# TreeNamedTuple uses the FIRST field's type -- fine for quantile's homogeneous numeric records.
_eltype(X::TreeArray)       = eltype(parent(X))
_eltype(X::TreeNamedTuple)  = _eltype(first(parent(X)))
_eltype(X::TreeRaggedArray) = _eltype(first(parent(X)))
_eltype(x)                  = eltype(x)


function Base.show(io::IO, T::Type{<:TreeDim})
    if get(io, :compact, false)
        print(io, name(T))
    else
        invoke(show, Tuple{IO, Type}, io, T)  # default
    end
end
function Base.show(io::IO, T::Type{<:TreeData})
    if get(io, :compact, false)
        print_type(io, T)
        print_dims(io, T)
    else
        invoke(show, Tuple{IO, Type}, io, T)  # default
    end
end

print_type(io::IO, ::Type{<:TreeData}) = print(io, "TreeData")
print_type(io::IO, ::Type{<:TreeNamedTuple}) = print(io, "TreeNamedTuple")
print_type(io::IO, ::Type{<:TreeRaggedArray}) = print(io, "TreeRaggedArray")
print_type(io::IO, ::Type{<:TreeArray}) = print(io, "TreeArray")
print_dims(io::IO, T::Type{<:TreeData}) = begin
    M = fieldtype(T, :meta)
    ds = fieldtypes(fieldtype(M, :dims))
    ds = :outer_dim in fieldnames(M) ? (ds..., fieldtype(M, :outer_dim)) : ds
    print(io, "("); join(io, ds, ", "); print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", x::TreeData) = show(io, x)

Base.show(io::IO, X::TreeData)  = print_tree(io, X)
print_tree(io::IO, X::TreeData) = begin
    compact = get(io, :compact, false)
    cio = IOContext(io, :compact => true)
    # @info get(io, :typeinfo, nothing)
    get(io, :typeinfo, Any) == typeof(X) ||  print(cio, typeof(X))
    compact && return print_values(cio, X)
    print(io, ":\n")
    print_dims(io, X)
    print_values(io, X)
end
print_dims(io::IO, X::TreeData) = begin
    print(io, "----------------\n")
    for dim in dims(X)
        print(io, dim, "\n")
    end
    haskey(meta(X), :outer_dim) && print(io, outerdim(X), "\n")
    print(io, "----------------")
end
print_values(io::IO, X::TreeData) = if get(io, :compact, false)
    print(io, parent(X))
else
    print(io, "\n")
    print(IOContext(io, :compact => true), parent(X))
end
print_values(io::IO, X::TreeNamedTuple) = if get(io, :compact, false)
    print(io, parent(X))
else
    print(io, "\n")
    io = IOContext(io, :compact => true)
    for (k, v) in pairs(parent(X))
        print(io, k, ": ", v, "\n")
    end
end
Base.show(io::IO, D::TreeDim) = begin
    print(io, name(D), ": ")
    print(IOContext(io, :compact => true), meta(D).values)
end

# ===================== dim helpers =====================
# an *axis* dim backs real structure (array axis / NT keys / ragged nesting): its value
# is `missing` (unlabelled) or a collection. A scalar value = a fixed-position singleton;
# `nothing` = an aggregated ("sliced") ghost. Neither backs an axis.
_isaxis(d::TreeDim) = _isaxis(meta(d).values)
_isaxis(::Union{Tuple,AbstractArray,AbstractRange}) = true   # array / NT / ragged axis
_isaxis(::Missing) = true                                    # unlabelled, but still an axis
_isaxis(_) = false                                           # scalar (fixed) or nothing (sliced)
_dimnames(d::Symbol) = (d,)
_dimnames(dims) = Tuple(dims)
_aschild(v::TreeData, inner) = v                 # already a TreeData -> knows its own dims
_aschild(v, inner) = TreeData(v, inner...)       # raw field -> wrap with the inner axes

# ===================== mapslices =====================
# Reduce the named `dims`: apply `f` to each leftover-index slice, keep everything.
# `f` returns a TreeData (or a scalar). Reduced dims stay but become `sliced` (aggregated).
# A requested dim that is absent from a leaf -> `missing` (fixed sentinel).
Base.mapslices(f, X::TreeData; dims) = error("mapslices: unhandled shape $(typeof(X))")

function Base.mapslices(f, X::TreeArray; dims)
    want    = _dimnames(dims)
    alldims = meta(X).dims
    names   = map(name, alldims)
    n_ax    = ndims(parent(X))
    redaxes = Tuple(i for i in 1:n_ax if names[i] in want)
    if isempty(redaxes)
        any(nm -> nm in names, want) || return missing               # dim absent here -> sentinel
        return TreeData(parent(X), (;dims = map(d -> name(d) in want ? sliced(d) : d, alldims)))
    end
    keepaxes = Tuple(i for i in 1:n_ax if !(names[i] in want))
    keptdims = Tuple(alldims[i] for i in keepaxes)
    keptset  = Set(keepaxes)
    trailing = Tuple(name(d) in want ? sliced(d) : d for (i, d) in enumerate(alldims) if !(i in keptset))
    outs = isempty(keepaxes) ? f(parent(X)) : map(f, eachslice(parent(X); dims=keepaxes))
    _assemble(outs, keptdims, trailing)
end

function Base.mapslices(f, X::TreeNamedTuple; dims)
    want = _dimnames(dims)
    rec  = outerdim(X)
    name(rec) in want && error("reducing the record dim $(name(rec)) is not supported")
    inner  = Tuple(d for d in meta(X).dims if _isaxis(d))       # outer no longer lives in dims
    ghosts = Tuple(d for d in meta(X).dims if !_isaxis(d))
    newfields = map(v -> mapslices(f, _aschild(v, inner); dims), parent(X))
    any(!ismissing, newfields) || return missing        # no child carried the dim -> sentinel
    sample = first(v for v in newfields if !ismissing(v))
    have   = map(name, meta(sample).dims)
    extra  = Tuple(g for g in ghosts if !(name(g) in have))
    TreeData(newfields, (;dims = (meta(sample).dims..., extra...), outer_dim = rec))
end

function Base.mapslices(f, X::TreeRaggedArray; dims)
    want = _dimnames(dims)
    any(name(d) in want for d in meta(X).dims) && error("reducing a ragged outer axis is not supported")
    TreeData(map(el -> mapslices(f, el; dims), parent(X)), meta(X))
end

# assemble the f-outputs into one TreeData, dispatching on the output shape. `outs` is a
# single output (reduce-all) or an array of outputs over the kept axes; `_over` bridges both.
_assemble(outs, keptdims, trailing) = _assemble(_proto(outs), outs, keptdims, trailing)
_proto(outs::AbstractArray) = first(outs)
_proto(out) = out
_over(g, outs::AbstractArray) = map(g, outs)   # preserve the kept-axes shape
_over(g, out) = g(out)                          # reduce-all: a single output

# record output -> structure of arrays: each field becomes its own array over the kept axes
function _assemble(proto::TreeNamedTuple, outs, keptdims, trailing)
    ks = keys(parent(proto))
    arrs = map(k -> _over(o -> parent(o)[k], outs), ks)
    TreeData(NamedTuple{ks}(arrs), (;dims = (keptdims..., trailing...), outer_dim = outerdim(proto)))
end
# new named axis (e.g. quantile levels) -> stack the output parents along that axis
_assemble(proto::TreeData, outs, keptdims, trailing) =
    TreeData(_stacklast(outs), (;dims = (keptdims..., dims(proto)..., trailing...)))
# scalar output -> a dense array of the scalars (`outs` already holds them)
_assemble(::Any, outs, keptdims, trailing) =
    TreeData(outs, (;dims = (keptdims..., trailing...)))

# stack the output parents along a new trailing axis, extracting parent inline (no temp
# array of parents); a lone reduce-all output passes its parent vector through as the axis.
_stacklast(out::TreeData) = parent(out)
function _stacklast(outs::AbstractArray)
    v1 = parent(first(outs))
    A  = Array{eltype(v1)}(undef, size(outs)..., length(v1))
    for I in CartesianIndices(outs)
        A[I, :] .= parent(outs[I])
    end
    A
end

# ===================== reducers =====================
Statistics.mean(X::TreeData; dims=nothing) = isnothing(dims) ? mean(parent(X)) : mapslices(mean, X; dims)
Base.sum(X::TreeData; dims) = mapslices(sum, X; dims)

# quantile delegates to mapslices; the output levels land on a *named, specifiable* axis
# (`into=`) so population- and posterior-quantiles can coexist. One shared sort buffer.
function Statistics.quantile(X::TreeData, p; dims, into = Symbol(only(_dimnames(dims)), :_quantile))
    levels   = collect(p)
    leveldim = TreeDim(into, Tuple(levels))   # constant across slices -> build once
    scratch  = _eltype(X)[]
    mapslices(X; dims) do slice
        length(scratch) == length(slice) || resize!(scratch, length(slice))
        copyto!(scratch, slice)
        TreeData(quantile!(scratch, levels), leveldim)
    end
end

# scalar p (e.g. quantile(X, 0.5; dims=:draw)) mirrors Base: each slice reduces to a
# SCALAR (no new axis, same scalar-output path `mean` exercises through mapslices);
# the requested level lands as a fixed-coordinate `into` dim (a scalar TreeDim value).
function Statistics.quantile(X::TreeData, p::Number; dims, into = Symbol(only(_dimnames(dims)), :_quantile))
    scratch = _eltype(X)[]
    result = mapslices(X; dims) do slice
        length(scratch) == length(slice) || resize!(scratch, length(slice))
        copyto!(scratch, slice)
        quantile!(scratch, p)
    end
    TreeData(result, TreeDim(into, p))
end

# ===================== dimension-aware kernels =====================
# Annotate a per-slice kernel with its dim-signature `reduces => into`. Two forms:
#   defining:  @kernel (:time => :stat) function f(L) ... end   -- define the plain array
#              kernel AND a TreeData method that threads it through mapslices.
#   post hoc:  @kernel (:time => :stat) f                        -- just add the TreeData
#              method to an already-defined f (e.g. a fixed-signature function you reuse).
# The output structure is inferred from the kernel's return type (the _assemble dispatch);
# the annotation only supplies the consumed dim + the record name. Parameterized reducers
# (mean/sum reduce->scalar, quantile reduce->axis, dim/into chosen per-call) DON'T fit this
# fixed-signature shape -- they stay as mapslices-delegating methods with dims=/into=.
macro kernel(spec, fdef)
    reduces = spec.args[2]                                   # consumed dim, e.g. :time
    into    = spec.args[3]                                   # produced record axis, e.g. :stat
    name    = fdef isa Symbol ? fdef : fdef.args[1].args[1]  # bare name (post hoc) or a def
    treemethod = :($name(X::TreeData) = mapslices(s -> TreeData($into => $name(s)), X; dims=$reduces))
    # a def -> also emit the plain array kernel; a bare name -> just wrap an existing function.
    esc(fdef isa Symbol ? treemethod : Expr(:block, fdef, treemethod))
end

unsetdim(X) = X
setdim(X::TreeData; kwargs...) = X#TreeData(unsetdim(parent(X); kwargs...), (;dims=setdim(meta(X).dims; kwargs...)))
setdim(dims::Tuple; kwargs...) = error()#values(merge(), (;kwargs...))

Base.cat(X::TreeData...) = TreeData(X)
Base.stack(f, iter::TreeDim) = map(f, iter)#
Base.stack(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{<:TreeDim}}}) = map(f, iter)
Base.map(f, iter::TreeDim) = TreeData(map(f, meta(iter).values), iter)
Base.map(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{<:TreeDim}}}) = TreeData(
    map(f, Iterators.product((meta(iter).values for iter in iter.iterators)...)), iter.iterators...
)


begin
    # # Should actually be doing some stuff to the X matrix - not doing it here for now
    # zero_re(X::TreeData) = setdim(X; random_effect=:zero)
    # # Should actually be doing some stuff to the X matrix - not doing it here for now, but maybe this will be the first change
    # constrain(X::TreeData) = setdim(X, dims.space.user)
    # # Should actually be doing some stuff to the X matrix - not doing it here for now
    # noplacebo(X::TreeData) = setdim(X, dims.placebo.off)
    # # Should actually be doing some stuff to the X matrix - not doing it here for now
    # setschedule(X::TreeData, schedule) = setdim(X, dims.schedule=>schedule)
    dense_loc(args...) = TreeData(
        randn(n_draws, n_subjects, n_dense),
        :draw, :subject, :time=>range(0, 1, n_dense)
    )
    # compute_stats: a dimension-aware kernel. Written once as pure math on a time-slice,
    # annotated `:time => :stat` -> also gets a TreeData method that reduces :time into a
    # :stat record. `compute_stats(::AbstractArray)` and `compute_stats(::TreeData)` both work.
    @kernel (:time => :stat) function compute_stats(L)
        trough, peak = extrema(L)
        baseline = L[1]
        dtrough, dpeak = extrema(L .- baseline)
        (;trough, peak, baseline, dtrough, dpeak)
    end


    n_subjects = 179
    healthy = rand(Bool, n_subjects)
    diseased = .!healthy
    male = rand(Bool, n_subjects)
    female = .!male
    weight = randn(n_subjects)
    age = randn(n_subjects)
    n_doses = [rand(1:28) for _ in 1:n_subjects]
    dose_times = map(sort ∘ randn, n_doses)
    dose_amounts = map(randn, n_doses)
    n_measurements = [rand(2:100) for _ in 1:n_subjects]
    measurement_times = map(sort ∘ randn, n_measurements)
    measurement_values = map(randn, n_measurements)
    # input_data = (;
    #     healthy, male, weight, age,
    #     dose_times, dose_amounts,
    #     measurement_times, measurement_values
    # )
    input_data = TreeData(
        :data=>(;
            healthy, diseased, male, female, weight, age,
            dose=map(TreeData(:time), dose_amounts, dose_times),
            measurement=map(TreeData(:time), measurement_values, measurement_times)
        ),
        :subject=>1:n_subjects
    )
    display(input_data)

    # reduce over :time through the heterogeneous, ragged tree: fields lacking a
    # :time axis come back `missing`; the ragged dose/measurement series are reduced.
    display(mapslices(mean, input_data; dims=:time))

    n_draws = 1000
    n_dense = 100
    sweep_schedules = TreeDim(:schedule, ("some schedule", ))
    sweep_doses = TreeDim(:dose, (20, 200))
    population_quantiles = (0.05, 0.5, 0.95)   # placeholder levels: summarize across subjects
    posterior_quantiles = (0.05, 0.5, 0.95)    # placeholder levels: summarize across draws

    n_params = (;
        baseline=6+n_subjects,
        dose=6+n_subjects,
        placebo=3,
        effect=4,
        noise=1,
        other=2
    )
    n_cols = sum(n_params)

    input_draws = TreeData(
        randn(n_draws, n_cols),
        :draw, :param; random_effect=:in_sample, placebo=:on, space=:sampler
    )
    stats_percentiles = map(Iterators.product(
        TreeDim(:random_effect, (:zero, :population)),
        TreeDim(:placebo, (:on, :off)),
        sweep_schedules,
        sweep_doses
    )) do args...
        quantile(
            quantile(
                compute_stats(dense_loc(input_draws, args...)),
                population_quantiles; dims=:subject, into=:population
            ),
            posterior_quantiles; dims=:draw, into=:posterior
        )
    end

    display(stats_percentiles)

    # demo: scalar-p quantile mirrors Base — `into` becomes a fixed SCALAR
    # coordinate (not a length-1 axis), and each slice holds one scalar value
    # (not a length-1 vector).
    median_draws = quantile(input_draws, 0.5; dims=:draw, into=:median)
    display(median_draws)
end
# begin



#     n_timepoints = 100
#     n_doses = 12

#     dims = map()

#     dims = (;
#         draw=(;
#             groups=(;
#                 chains=1,
#             )
#         ),
#         param=(;
#             groups=(;
#                 baseline=(;
#                     fixed=(;
#                         intercept=1,
#                         zage=2,
#                         zweight=3,
#                         male=4,
#                         diseased=5,
#                     ),
#                     log_scale=6,
#                     random=7:6+n_subjects
#                 )
#             )
#         ),
#         time=(;
#             # value=range(0, 1, n_draws),
#         ),
#         schedule=(;

#         ),
#         dose=(;
#             value=exp.(range(0, 1, n_doses))
#         ),
#         subject=(;
#             value=1:n_subjects,
#             groups=(;healty, diseased, male, female)
#         ),
#         random_effect=(;
#             value=(:in_sample, :zero, :population)
#         ),
#         placebo=(;
#             value=(:on, :off)
#         ),
#         space=(;value=(:sampler, :user))
#     )

#     S = TreeData(
#         randn(n_draws, n_cols),
#         (;dims=(
#             dims.draw,
#             dims.param,
#             dims.random_effect.in_sample,
#             dims.placebo.on,
#             dims.space.sampler
#         ))
#     )
#     S0 = zero_re(S)
#     SS0 = stack((S, S0))
#     UU0 = constrain(SS0)
#     R2 = mapslices(UU0; dims=(:subject, :random_effect)) do sub
#         tmp = var(sub; dims=:subject)
#         tmp[tmp.random_effect.zero] ./ tmp[tmp.random_effect.in_sample]
#     end



#     P = setschedule(setre(S, :population), "something")
#     Ls = stack(Iterators.product(doses, placebos)) do dose, placebo
#         loc(setplacebo(setdose(P, dose), placebo))
#     end

#     stats = compute_stats(Ls)
#     stats_percentiles = quantile(quantile(stats, population_quantiles; dims=subject), posterior_quantiles; dims=draw)

#     dLs = mapslices(Ls; dims=time) do L
#         setdim(L .- L[1]; qoi="d" * L.qoi)
#     end
#     LdLs = stack((Ls, dLs))
#     LdLs_percentiles = quantile(quantile(LdLs, population_quantiles; dims=subject), posterior_quantiles; dims=draw)



# end
