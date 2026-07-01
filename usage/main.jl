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
Base.length(X::TreeDim) = length(meta(X).values) 
Base.iterate(X::TreeDim, args...) = iterate(meta(X).values, args...)
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
        values=keys(X),
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

Base.mapslices(f, X::TreeData; dims) = error()
Base.mapslices(f, X::TreeArray; dims) = begin 
    dims = isa(dims, Symbol) ? (dims,) : dims
    if dims == map(name, meta(X).dims)
        TreeData(f(X), (;dims=map(sliced, meta(X).dims)))
    else
        @warn "Skipping $(dims=>map(name, meta(X).dims))"
    end
end
Base.mapslices(f, X::TreeRaggedArray; dims) = begin 
    dims = isa(dims, Symbol) ? (dims,) : dims
    for dim in meta(X).dims
        @assert name(dim) ∉ dims
    end
    TreeData(map(x->mapslices(f, TreeData(x, Base.front(meta(X).dims)...); dims), parent(X)), meta(X))
end
Base.mapslices(f, X::TreeNamedTuple; dims) = begin
    dims = isa(dims, Symbol) ? (dims,) : dims
    for dim in meta(X).dims
        @assert name(dim) ∉ dims
    end
    TreeData(map(x->mapslices(f, TreeData(x, Base.front(meta(X).dims)...); dims), parent(X)), meta(X))
end

Statistics.mean(X::TreeData) = mean(parent(X))

unsetdim(X) = X
setdim(X::TreeData; kwargs...) = X#TreeData(unsetdim(parent(X); kwargs...), (;dims=setdim(meta(X).dims; kwargs...)))
setdim(dims::Tuple; kwargs...) = error()#values(merge(), (;kwargs...))
# Statistics.mean(X::TreeData; kwargs...) = mapslices(mean, X; kwargs...)
# Statistics.quantile(X::TreeData, p; kwargs...) = mapslices(Base.Fix2(quantile, p), X; kwargs...)

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
    dense_loc(args...) = TreeArray(
        randn(n_draws, n_subjects, n_dense),
        :draw, :subject, :time=>range(0, 1, n_dense)
    )
    compute_stats(Ls) = mapslices(Ls; dims=:time) do L
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
    # # view(input_data; data=:healthy)
    # display(mapslices(mean, input_data; dims=:time))

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
                population_quantiles; dims=:subject
            ), 
            posterior_quantiles; dims=:draw
        )
    end

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

