# Option B — a MOCK standalone TreeArrays (comparison artifact, NOT src/).
#
# Demonstrates the two deliberate differentiators (REQUIREMENTS.md §2):
#   1. Lazy/explicit ASSEMBLY, not lazy compute. `stack` is STRUCTURAL — it
#      holds references to its blocks and NEVER allocates the big combined
#      matrix. The reduction streams: materialize one block → eager kernel →
#      discard → next. (Contrast: DD's `cat` materializes the whole thing.)
#   2. Native RAGGED axes via CSR offsets — exact contiguous storage, zero
#      padding. The SAME `mapslices`-style code runs on it. (Contrast: DD must
#      pad to a rectangle.)
#
# Everything here is eager arithmetic in tight loops; nothing is a deferred
# compute graph. This is a throwaway sketch to judge ergonomics, not the real
# package.
module OptionB

using Statistics
using ..BenchKit

# ── Slice 1: structural assembly + streaming reduction ──────────────────────

"A draws block with a fixed `random_effect` coordinate. Wraps the matrix; no copy."
struct Block
    data::Matrix{Float64}
    re_cols::UnitRange{Int}
end

"Wrap draws as a block fixed at `random_effect = in_sample`. O(1), no copy."
construct(draws, re_cols) = Block(draws, re_cols)

"random_effect.zero: zero the random-effect columns (the real denotation of the
sketch's no-op). Allocates one block — the differentiator is the STACK, not this."
function zero_re(b::Block)
    d = copy(b.data)
    @views d[:, b.re_cols] .= 0.0
    Block(d, b.re_cols)
end

"""
Structural stack along a new `random_effect` axis. Holds block REFERENCES and
the level labels — it does NOT allocate the combined `n_re × n_draws × n_cols`
array. That cross-product axis is described, never materialized.
"""
struct StructuralStack
    blocks::Vector{Block}
    levels::Vector{Symbol}
end
stack(blocks::Tuple, levels) = StructuralStack(collect(blocks), levels)

"""
    var_ratio(ss)

Per-column variance over draws for each block, STREAMED: one block materialized
at a time (its var computed, then discarded) — the combined matrix is never
held. Returns `var(zero) ./ var(in_sample)` per column (the shrinkage/R²-style
quantity): ≈0 on the zeroed random-effect columns, ≈1 elsewhere.
"""
function var_ratio(ss::StructuralStack)
    per_block = map(ss.blocks) do b
        vec(var(b.data; dims=1))          # eager kernel on ONE block, then discard
    end
    per_block[2] ./ per_block[1]          # zero / in_sample
end

# ── Slice 2: native ragged axis (CSR) + mapslices ───────────────────────────

"Ragged axis as CSR: contiguous `values`, `offsets[i]+1:offsets[i+1]` = group i.
Exact storage, no padding. The identity-bearing structure co-indexed columns share."
struct CSR
    values::Vector{Float64}
    offsets::Vector{Int}                  # length n_groups + 1
end

function csr(per_group::Vector{Vector{Float64}})
    offsets = cumsum(vcat(0, length.(per_group)))
    values  = reduce(vcat, per_group)
    CSR(values, offsets)
end

n_groups(r::CSR) = length(r.offsets) - 1
group_view(r::CSR, i) = @view r.values[r.offsets[i] + 1 : r.offsets[i + 1]]

"`mapslices(f; dims=:subject)` on a ragged axis: f gets one contiguous, EXACTLY-
sized per-group view. No padding, no skipmissing — the same f a rectangular
input would use."
mapslices_ragged(f, r::CSR) = map(i -> f(group_view(r, i)), 1:n_groups(r))

"compute_stats — SAME code on ragged as it would be on rectangular (invariant §7)."
compute_stats(r::CSR) = mapslices_ragged(r) do L
    trough, peak   = extrema(L)
    baseline       = L[1]
    dtrough, dpeak = extrema(L .- baseline)
    (; trough, peak, baseline, dtrough, dpeak)
end

# ── Driver: run every step, benchmark each (time + bytes), return results ────

function run(data)
    (; draws, re_cols, measurement_values) = data

    S   = construct(draws, re_cols)
    S0  = zero_re(S)
    SS0 = stack((S, S0), [:in_sample, :zero])
    ratio = var_ratio(SS0)
    rag   = csr(measurement_values)
    stats = compute_stats(rag)
    pquant = [0.1, 0.5, 0.9]
    troughs = [s.trough for s in stats]
    cross = quantile(troughs, pquant)

    steps = [
        BenchKit.step("construct S", () -> construct(draws, re_cols);
            note="wrap draws, O(1) no copy"),
        BenchKit.step("zero_re → S0", () -> zero_re(S);
            note="copy + zero RE columns"),
        BenchKit.step("stack((S, S0))", () -> stack((S, S0), [:in_sample, :zero]);
            note="STRUCTURAL — holds refs, combined matrix NEVER allocated"),
        BenchKit.step("var ratio (streamed)", () -> var_ratio(SS0);
            note="one block materialized at a time, then discarded"),
        BenchKit.step("ragged compute_stats", () -> compute_stats(csr(measurement_values));
            note="CSR build + mapslices over exact contiguous views, no padding"),
        BenchKit.step("cross-subject quantile", () -> quantile([s.trough for s in stats], pquant);
            note="re-densified per-subject trough → quantile"),
    ]

    # RE-column ratios should be ≈0, others ≈1 — summarize honestly.
    re = collect(re_cols)
    other = setdiff(1:length(ratio), re)
    outputs = (;
        ratio_re_mean    = mean(ratio[re]),
        ratio_other_mean = mean(ratio[other]),
        n_re             = length(re),
        n_other          = length(other),
        stats_sample     = stats[1:3],
        cross_quantiles  = cross,
        ragged_n_values  = length(rag.values),
        ragged_padded_n  = n_groups(rag) * maximum(diff(rag.offsets)),
    )

    (; steps, outputs)
end

end # module OptionB
