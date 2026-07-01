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
TreeData((name, X)::Pair{Symbol,<:NamedTuple}, dims::TreeDim...) = TreeData(
    X, dims..., TreeDim(name, (;
        values=keys(X), outer=true,
    ))
)
TreeData(X::TreeData, dims::TreeDim...) = TreeData(parent(X), meta(X).dims..., dims...)
TreeNamedTuple{P<:NamedTuple,M<:NamedTuple} = TreeData{P,M}
TreeRaggedArray{P<:AbstractArray{<:TreeData},M<:NamedTuple} = TreeData{P,M}
TreeArray{P<:AbstractArray,M<:NamedTuple} = TreeData{P,M}


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
    print(io, "(")
    join(io, fieldtypes(fieldtype(fieldtype(T, :meta), :dims)), ", ")
    print(io, ")")
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
    for dim in meta(X).dims
        print(io, dim, "\n")
    end
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
_isouter(d::TreeDim) = get(meta(d), :outer, false)
_outerdim(X::TreeData) = only(d for d in meta(X).dims if _isouter(d))
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
    rec  = _outerdim(X)
    name(rec) in want && error("reducing the record dim $(name(rec)) is not supported")
    inner  = Tuple(d for d in meta(X).dims if _isaxis(d) && !_isouter(d))
    ghosts = Tuple(d for d in meta(X).dims if !_isaxis(d) && !_isouter(d))
    newfields = map(v -> mapslices(f, _aschild(v, inner); dims), parent(X))
    any(!ismissing, newfields) || return missing        # no child carried the dim -> sentinel
    sample = first(v for v in newfields if !ismissing(v))
    have   = map(name, meta(sample).dims)
    extra  = Tuple(g for g in ghosts if !(name(g) in have))
    TreeData(newfields, (;dims = (meta(sample).dims..., extra..., rec)))
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
    TreeData(NamedTuple{ks}(arrs), (;dims = (keptdims..., trailing..., _outerdim(proto))))
end
# new named axis (e.g. quantile levels) -> stack the output parents along that axis
_assemble(proto::TreeData, outs, keptdims, trailing) =
    TreeData(_stacklast(_over(parent, outs)), (;dims = (keptdims..., meta(proto).dims..., trailing...)))
# scalar output -> a dense array of the scalars (`outs` already holds them)
_assemble(::Any, outs, keptdims, trailing) =
    TreeData(outs, (;dims = (keptdims..., trailing...)))

# stack equal-length vectors along a new trailing axis; a lone axis-vector passes through
_stacklast(v::AbstractVector{<:Number}) = v
function _stacklast(vs::AbstractArray)
    n = length(first(vs))
    A = Array{eltype(first(vs))}(undef, size(vs)..., n)
    for I in CartesianIndices(vs)
        A[I, :] .= vs[I]
    end
    A
end

# ===================== reducers =====================
Statistics.mean(X::TreeData; dims=nothing) = isnothing(dims) ? mean(parent(X)) : mapslices(mean, X; dims)
Base.sum(X::TreeData; dims) = mapslices(sum, X; dims)

# quantile delegates to mapslices; the output levels land on a *named, specifiable* axis
# (`into=`) so population- and posterior-quantiles can coexist. One shared sort buffer.
function Statistics.quantile(X::TreeData, p; dims, into = Symbol(only(_dimnames(dims)), :_quantile))
    levels  = collect(p)
    scratch = Float64[]
    mapslices(X; dims) do slice
        n = length(slice)
        length(scratch) == n || resize!(scratch, n)
        copyto!(scratch, slice)
        TreeData(quantile!(scratch, levels), TreeDim(into, Tuple(levels)))
    end
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
    compute_stats(Ls) = mapslices(Ls; dims=:time) do L
        trough, peak = extrema(L)
        baseline = L[1]
        dtrough, dpeak = extrema(L .- baseline)
        TreeData(:stat => (;trough, peak, baseline, dtrough, dpeak))
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
