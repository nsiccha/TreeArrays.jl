# Option A — the SAME slice implemented with DimensionalData.jl (comparison
# artifact, NOT src/).
#
# DD gives us named dims, dim-aware reductions, and `dims=:name` ergonomics for
# free — the parts TreeArrays would happily reuse. Where it shows friction
# (REQUIREMENTS.md §2) is exactly the two differentiators:
#   1. ASSEMBLY is eager + materializing. `cat` allocates the full combined
#      `n_draws × n_cols × n_re` block (~6 MB here) — there is no structural /
#      streamed assembly; DD is eager.
#   2. RAGGED data is unsupported — DD is fundamentally rectangular, so the
#      per-subject series must be PADDED to a `subject × time` matrix with
#      `missing`, wasting storage AND forcing skipmissing in the per-group
#      kernel (DD's named `mapslices` won't cleanly do a per-group NamedTuple
#      reduction over the padded rectangle, so we drop to `eachrow(parent(...))`).
#
# This is the idiomatic, FAIR DD path (decision q6vafs, option A) — not a
# strawman. The `Vector{DimArray}` alternative avoids padding waste but loses
# cross-subject dim ops; noted in the benchmark verdict.
module OptionA

using DimensionalData
using Statistics
using ..BenchKit

# ── Slice 1: eager, materializing assembly ──────────────────────────────────

"Wrap draws as a named DimArray (draw × param). O(1), like the mock's wrap."
construct(draws) =
    DimArray(draws, (Dim{:draw}(1:size(draws, 1)), Dim{:param}(1:size(draws, 2))))

"random_effect.zero: zero the random-effect columns. Allocates one block."
function zero_re(S, re_cols)
    d = copy(parent(S))
    @views d[:, re_cols] .= 0.0
    DimArray(d, dims(S))
end

"""
DD assembly = `cat` along a NEW `random_effect` dim. This MATERIALIZES the full
combined `draw × param × random_effect` array (~6 MB) — DD is eager, there is no
structural/lazy assembly. This single call is where the allocation gap lives.
"""
stack(S, S0) = cat(S, S0; dims=Dim{:random_effect}([:in_sample, :zero]))

"""
    var_ratio(SS0)

`var(...; dims=:draw)` — DD's named-dim reduction (the nice part). Then the
shrinkage ratio zero/in_sample per param.
"""
function var_ratio(SS0)
    v = var(SS0; dims=:draw)               # → draw reduced to length 1
    vp = dropdims(parent(v); dims=1)       # param × random_effect
    vp[:, 2] ./ vp[:, 1]                    # zero / in_sample
end

# ── Slice 2: ragged forced into a padded rectangle ──────────────────────────

"""
Pad the ragged per-subject series to a `subject × time` DimArray with `missing`.
This is the workaround DD forces: storage is `Union{Missing,Float64}` over the
MAX length (waste vs the exact CSR), and every downstream kernel must skip the
padding.
"""
function pad(per_subject::Vector{Vector{Float64}})
    n = length(per_subject)
    maxlen = maximum(length, per_subject)
    M = Matrix{Union{Missing,Float64}}(missing, n, maxlen)
    for (i, v) in enumerate(per_subject)
        M[i, 1:length(v)] .= v
    end
    DimArray(M, (Dim{:subject}(1:n), Dim{:time}(1:maxlen)))
end

"""
compute_stats over the padded rectangle. DD's named `mapslices` won't cleanly
collect a per-group NamedTuple reduction over padded rows, so we drop to
`eachrow(parent(...))` + `skipmissing` — the ergonomic friction the mock avoids.
"""
function compute_stats(A)
    map(eachrow(parent(A))) do row
        L = collect(skipmissing(row))      # strip the padding every time
        trough, peak   = extrema(L)
        baseline       = L[1]
        dtrough, dpeak = extrema(L .- baseline)
        (; trough, peak, baseline, dtrough, dpeak)
    end
end

# ── Driver: same steps as Option B, benchmarked identically ─────────────────

function run(data)
    (; draws, re_cols, measurement_values) = data

    S   = construct(draws)
    S0  = zero_re(S, re_cols)
    SS0 = stack(S, S0)
    ratio = var_ratio(SS0)
    padded = pad(measurement_values)
    stats  = compute_stats(padded)
    pquant = [0.1, 0.5, 0.9]
    troughs = [s.trough for s in stats]
    cross = quantile(troughs, pquant)

    steps = [
        BenchKit.step("construct S", () -> construct(draws);
            note="wrap draws in DimArray, O(1)"),
        BenchKit.step("zero_re → S0", () -> zero_re(S, re_cols);
            note="copy + zero RE columns"),
        BenchKit.step("stack((S, S0))", () -> stack(S, S0);
            note="cat MATERIALIZES the full combined block (~6 MB)"),
        BenchKit.step("var ratio (eager)", () -> var_ratio(SS0);
            note="var(;dims=:draw) over the materialized block"),
        BenchKit.step("ragged compute_stats", () -> compute_stats(pad(measurement_values));
            note="PAD to subject×time missing-rectangle + skipmissing per row"),
        BenchKit.step("cross-subject quantile", () -> quantile([s.trough for s in stats], pquant);
            note="re-densified per-subject trough → quantile"),
    ]

    re = collect(re_cols)
    other = setdiff(1:length(ratio), re)
    outputs = (;
        ratio_re_mean    = mean(ratio[re]),
        ratio_other_mean = mean(ratio[other]),
        n_re             = length(re),
        n_other          = length(other),
        stats_sample     = stats[1:3],
        cross_quantiles  = cross,
        ragged_n_values  = sum(length, measurement_values),
        ragged_padded_n  = length(measurement_values) * maximum(length, measurement_values),
    )

    (; steps, outputs)
end

end # module OptionA
