# # TreeArrays PKPD demo
#
# A literate walkthrough of TreeArrays' reduction pipeline on a synthetic
# PKPD (pharmacokinetic/pharmacodynamic) dataset: ragged per-subject
# measurement/dose histories, a dense posterior-draw sweep across scenarios,
# and the `(:population, :posterior)` quantile-matrix summary a PKPD report
# actually consumes.
#
# Runs at demo scale (179 subjects x 1000 draws x 100 time points) -- this
# is the same scale the design was verified against, not a shrunk unit test.
# Expect ~15s / ~1GB.

using TreeArrays

# ## A dimension-aware kernel
#
# `compute_stats` is written once as plain math on a time-slice. Annotating
# it `@kernel (:time => :stat)` also derives a `TreeData` method that reduces
# the named `:time` axis into a `:stat` record -- `compute_stats` works on
# both a bare `AbstractArray` and a `TreeData`.
@kernel (:time => :stat) function compute_stats(L)
    trough, peak = extrema(L)
    baseline = L[1]
    dtrough, dpeak = extrema(L .- baseline)
    (;trough, peak, baseline, dtrough, dpeak)
end

# ## Ragged per-subject input data
#
# Each subject has a different number of doses and measurements -- a ragged
# tree of `TreeData` leaves under one outer `:subject` axis.
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

input_data = TreeData(
    :data=>(;
        healthy, diseased, male, female, weight, age,
        dose=map(TreeData(:time), dose_amounts, dose_times),
        measurement=map(TreeData(:time), measurement_values, measurement_times)
    ),
    :subject=>1:n_subjects
)
display(input_data)

# Reducing over `:time` walks the heterogeneous, ragged tree: fields lacking
# a `:time` axis (e.g. `healthy`, `weight`) come back `missing`; the ragged
# `dose`/`measurement` series are reduced per-subject.
display(mapslices(mean, input_data; dims=:time))

# ## Posterior draws + a scenario sweep
#
# `dense_loc` stands in for the (upstream, out-of-scope) ODE solve that maps
# posterior draws to a dense prediction curve over a `:time` grid.
n_draws = 1000
n_dense = 100

n_params = (;
    baseline=6+n_subjects,
    dose=6+n_subjects,
    placebo=3,
    effect=4,
    noise=1,
    other=2
)
n_cols = sum(n_params)

dense_loc(args...) = TreeData(
    randn(n_draws, n_subjects, n_dense),
    :draw, :subject, :time=>range(0, 1, n_dense)
)

input_draws = TreeData(
    randn(n_draws, n_cols),
    :draw, :param; random_effect=:in_sample, placebo=:on, space=:sampler
)

sweep_schedules = TreeDim(:schedule, ("some schedule", ))
sweep_doses = TreeDim(:dose, (20, 200))
population_quantiles = (0.05, 0.5, 0.95)   # placeholder levels: summarize across subjects
posterior_quantiles = (0.05, 0.5, 0.95)    # placeholder levels: summarize across draws

# For each `(random_effect, placebo, schedule, dose)` scenario: compute
# per-time-point stats, then quantile across subjects (`:population`) and
# across draws (`:posterior`) -- a chained reduction, never eagerly
# restructured.
stats_percentiles = map(Iterators.product(
    TreeDim(:random_effect, (:zero, :population)),
    TreeDim(:placebo, (:on, :off)),
    sweep_schedules,
    sweep_doses
)) do args...
    quantile(
        quantile(
            compute_stats(dense_loc(input_draws, args...)),
            TreeDim(:population, population_quantiles); dims=:subject
        ),
        TreeDim(:posterior, posterior_quantiles); dims=:draw
    )
end

display(stats_percentiles)

# ## Scalar-p quantile
#
# A scalar `p` mirrors Base's `quantile` convention: the pdim's fixed
# SCALAR value becomes a fixed coordinate (not a length-1 axis), and each
# slice holds one scalar value (not a length-1 vector).
median_draws = quantile(input_draws, TreeDim(:median, 0.5); dims=:draw)
display(median_draws)

# ## Tables.jl export
#
# Both chained-reduction results above are, unmodified, Tables.jl COLUMN
# sources -- no separate export step, no DataFrame in between. Nothing melts
# at construction (everything above stayed lazy); the tree -> columns melt
# happens exactly once, here, at `Tables.columns`.
using Tables

Tables.istable(stats_percentiles)   # true
sch = Tables.schema(stats_percentiles)   # computed from the TYPE alone, no melt
display(sch)

cols = Tables.columns(stats_percentiles)
display(Tables.columnnames(cols))
display(first(Tables.rowtable(stats_percentiles), 3))

display(Tables.columns(median_draws))
