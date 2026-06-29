using TreeArrays, DataFrames, LazyArrays, Statistics



struct TreeArray{T,N,P<:AbstractArray{T,N},M}<:AbstractArray{T,N}
    parent::P
    meta::M
end
Base.parent(X::TreeArray) = getfield(X, :parent)
meta(X::TreeArray) = getfield(X, :meta)

Base.mapslices(f, X::TreeArray; dims) = error()
Statistics.mean(X::TreeArray; kwargs...) = mapslices(mean, X; kwargs...)
Statistics.quantile(X::TreeArray, p; kwargs...) = mapslices(Base.Fix2(quantile, p), X; kwargs...)


begin 
    # Should actually be doing some stuff to the X matrix - not doing it here for now
    zero_re(X::TreeArray) = setdim(X, dims.random_effect.zero)
    # Should actually be doing some stuff to the X matrix - not doing it here for now, but maybe this will be the first change
    constrain(X::TreeArray) = setdim(X, dims.space.user)
    # Should actually be doing some stuff to the X matrix - not doing it here for now
    noplacebo(X::TreeArray) = setdim(X, dims.placebo.off)
    # Should actually be doing some stuff to the X matrix - not doing it here for now
    setschedule(X::TreeArray, schedule) = setdim(X, dims.schedule=>schedule)
    compute_stats(Ls) = mapslices(Ls; dims=time) do L 
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
    input_data = (;
        healthy, male, weight, age,
        dose_times, dose_amounts,
        measurement_times, measurement_values
    )



    n_params = (;
        baseline=6+n_subjects,
        dose=6+n_subjects,
        placebo=3,
        effect=4,
        noise=1,
        other=2
    )
    n_cols = sum(n_params)
    n_timepoints = 100
    n_draws = 1000
    n_doses = 12

    dims = (;
        draw=(;
            groups=(;
                chains=1,
            )
        ),
        param=(;
            groups=(;
                baseline=(;
                    fixed=(;
                        intercept=1,
                        zage=2,
                        zweight=3,
                        male=4,
                        diseased=5,
                    ),
                    log_scale=6,
                    random=7:6+n_subjects
                )
            )
        ),
        time=(;
            # value=range(0, 1, n_draws),
        ),
        schedule=(;

        ),
        dose=(;
            value=exp.(range(0, 1, n_doses))
        ),
        subject=(;
            value=1:n_subjects,
            groups=(;healty, diseased, male, female)
        ),
        random_effect=(;
            value=(:in_sample, :zero, :population)
        ),
        placebo=(;
            value=(:on, :off)
        ),
        space=(;value=(:sampler, :user))
    )

    S = TreeArray(
        randn(n_draws, n_cols),
        (;dims=(
            dims.draw, 
            dims.param, 
            dims.random_effect.in_sample, 
            dims.placebo.on, 
            dims.space.sampler
        ))
    )
    S0 = zero_re(S)
    SS0 = stack((S, S0))
    UU0 = constrain(SS0)
    R2 = mapslices(UU0; dims=(:subject, :random_effect)) do sub
        tmp = var(sub; dims=:subject)
        tmp[tmp.random_effect.zero] ./ tmp[tmp.random_effect.in_sample]
    end



    P = setschedule(setre(S, :population), "something") 
    Ls = stack(Iterators.product(doses, placebos)) do dose, placebo 
        loc(setplacebo(setdose(P, dose), placebo))
    end

    stats = compute_stats(Ls)
    stats_percentiles = quantile(quantile(stats, population_quantiles; dims=subject), posterior_quantiles; dims=draw)

    dLs = mapslices(Ls; dims=time) do L
        setdim(L .- L[1]; qoi="d" * L.qoi)
    end
    LdLs = stack((Ls, dLs))
    LdLs_percentiles = quantile(quantile(LdLs, population_quantiles; dims=subject), posterior_quantiles; dims=draw)



end

