# Minimal-but-representative benchmark of the "full draws -> plot-ready summary"
# reduction, compared across data-array/table stacks with Chairmarks `@be`.
#
# ONE pipeline scenario (user: keep pipelines minimal, start with a single one):
# a posterior-predictive POPULATION-MEAN TIMECOURSE RIBBON — the shape a
# `lineribbon` consumes (x = time survives as the ribbon's x-axis):
#
#   raw draws  [draw x subject x time]
#     -> mean over :subject   -> population-mean trajectory per (draw, time)
#     -> quantile over :draw   -> posterior ribbon bands per time
#   => a (time x band) summary  ->  lineribbon(x = time, bands = …)
#
# Backends (the "4" of the user's 4x2 matrix, + a plain-Array floor):
#   Array           - explicit loops; the "speed of light" floor.
#   DataFrames      - the incumbent combine(groupby) (must MELT the 3D array into
#                     an n_draw*n_subj*n_time long frame first).
#   DimensionalData - named-dim storage + EAGER reduce.
#   FlexiChains     - a sampler-output container (draws-in-a-chain); a dense
#                     prediction tensor is an awkward fit (its own docs: array-
#                     valued params must be broken up) — included as a round-trip.
#   TreeArrays      - chained cross-axis reduction; labels stay axes, nothing melts.
#
# Plot tail (AoV + AoG, the "x2") is staged separately (needs the Makie/AoV env);
# this file is the backend-reduction matrix. NaN-free synthetic data + plain
# type-7 quantile uniformly => identical numbers + apples-to-apples timing.
#
# Run:  julia --project=benchmark benchmark/benchmarks.jl

using TreeArrays                      # TreeData, TreeDim, mapslices, mean, quantile (bare = TA idiom)
using Chairmarks, Printf
import Statistics
import Tables
import DataFrames as DF
import DimensionalData as DD
import FlexiChains as FC

const N_DRAW = 1000
const N_SUBJ = 179
const N_TIME = 50
const BANDS  = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)   # posterior ribbon (across draws)
const BANDSV = collect(BANDS)

gen_draws() = randn(N_DRAW, N_SUBJ, N_TIME)   # [draw x subject x time]

# ---------------------------------------------------------------------------
# 0. Plain Array baseline — explicit loops, the "no abstraction" floor.
# ---------------------------------------------------------------------------
function pipeline_array(x)
    nd, ns, nt = size(x)
    m = Matrix{Float64}(undef, nd, nt)                     # population mean per (draw,time)
    @inbounds for t in 1:nt, d in 1:nd
        acc = 0.0
        for j in 1:ns
            acc += x[d, j, t]
        end
        m[d, t] = acc / ns
    end
    out = Matrix{Float64}(undef, nt, length(BANDS))        # ribbon per time
    @inbounds for t in 1:nt
        out[t, :] .= Statistics.quantile(@view(m[:, t]), BANDSV)
    end
    out
end

# ---------------------------------------------------------------------------
# 1. DataFrames — the incumbent combine(groupby) (+ the melt it forces).
# ---------------------------------------------------------------------------
function pipeline_df(x)
    nd, ns, nt = size(x)
    df = DF.DataFrame(
        draw    = repeat(1:nd, outer = ns * nt),
        subject = repeat(1:ns, inner = nd, outer = nt),
        time    = repeat(1:nt, inner = nd * ns),
        value   = vec(x),
    )
    g1 = DF.combine(DF.groupby(df, [:draw, :time]), :value => Statistics.mean => :m)
    g2 = DF.combine(DF.groupby(g1, :time)) do s
        DF.DataFrame(band = BANDSV, val = Statistics.quantile(s.m, BANDSV))
    end
    g2
end

# ---------------------------------------------------------------------------
# 2. DimensionalData — named-dim storage + EAGER reduce (mean over :subject is
#    DD's sweet spot; the ribbon quantile has no lazy new-axis form so it drops
#    back to manual array work).
# ---------------------------------------------------------------------------
function pipeline_dd(x)
    nd, ns, nt = size(x)
    A = DD.DimArray(x, (DD.Dim{:draw}(1:nd), DD.Dim{:subject}(1:ns), DD.Dim{:time}(1:nt)))
    m = dropdims(Statistics.mean(A; dims = DD.Dim{:subject}); dims = DD.Dim{:subject})  # draw × time
    mm = parent(m)
    out = Matrix{Float64}(undef, nt, length(BANDS))
    @inbounds for t in 1:nt
        out[t, :] .= Statistics.quantile(@view(mm[:, t]), BANDSV)
    end
    out
end

# ---------------------------------------------------------------------------
# 3. FlexiChains — draws held in a VNChain (one array-valued `pred` var), then
#    extracted + reduced. Represents "your draws arrived in a chains container".
# ---------------------------------------------------------------------------
function pipeline_flexi(x)
    nd, ns, nt = size(x)
    col = reshape([x[d, :, :] for d in 1:nd], nd, 1)       # (nd×1) of (ns×nt) matrices
    ch = FC.FlexiChain{FC.VarName}(nd, 1, Dict(FC.Parameter(FC.@varname(pred)) => col))
    dm = ch[FC.@varname(pred)]                              # DimMatrix (nd×1) of matrices
    m = Matrix{Float64}(undef, nd, nt)
    @inbounds for d in 1:nd
        pm = dm[d, 1]
        for t in 1:nt
            acc = 0.0
            for j in 1:ns
                acc += pm[j, t]
            end
            m[d, t] = acc / ns
        end
    end
    out = Matrix{Float64}(undef, nt, length(BANDS))
    @inbounds for t in 1:nt
        out[t, :] .= Statistics.quantile(@view(m[:, t]), BANDSV)
    end
    out
end

# ---------------------------------------------------------------------------
# 4. TreeArrays — chained cross-axis reduction; nothing is stacked/melted.
# ---------------------------------------------------------------------------
function pipeline_ta(x)
    nd, ns, nt = size(x)
    X = TreeData(x, :draw, :subject, :time => 1:nt)
    quantile(mean(X; dims = :subject),                     # -> draw × time (population mean)
             TreeDim(:band, BANDS); dims = :draw)          # -> time × band (posterior ribbon)
end

# ---------------------------------------------------------------------------
# Correctness: reduce every stack's output to the same (TIME x BANDS) matrix.
# ---------------------------------------------------------------------------
canon_array(out::Matrix) = out
canon_dd(out::Matrix) = out
canon_flexi(out::Matrix) = out

function canon_df(g2)
    out = Matrix{Float64}(undef, N_TIME, length(BANDS))
    for r in eachrow(g2)
        out[r.time, findfirst(≈(r.band), BANDSV)] = r.val
    end
    out
end

function canon_ta(result)
    names = Tables.columnnames(Tables.columns(result))
    @info "TreeArrays melted columns" names
    valname = only(setdiff(collect(names), (:time, :band)))
    out = Matrix{Float64}(undef, N_TIME, length(BANDS))
    for row in Tables.rows(result)
        out[Int(row.time), findfirst(≈(row.band), BANDSV)] = getproperty(row, valname)
    end
    out
end

const STACKS = [
    ("Array (floor)",   pipeline_array, canon_array),
    ("DataFrames",      pipeline_df,    canon_df),
    ("DimensionalData", pipeline_dd,    canon_dd),
    ("FlexiChains",     pipeline_flexi, canon_flexi),
    ("TreeArrays",      pipeline_ta,    canon_ta),
]

function check_equivalence()
    x = gen_draws()
    ref = canon_array(pipeline_array(x))
    println("output-equivalence vs Array floor:")
    for (name, f, canon) in STACKS
        try
            m = canon(f(x))
            @printf("  %-16s %s (max |Δ| = %.2e)\n", name, m ≈ ref ? "✓" : "✗ MISMATCH",
                    maximum(abs, m .- ref))
        catch e
            @printf("  %-16s ⚠ errored: %s\n", name, sprint(showerror, e))
        end
    end
end

function run_benchmarks()
    println("\nscale: $N_DRAW draws x $N_SUBJ subjects x $N_TIME time  (@be, min sample)\n")
    @printf("%-16s %12s %12s %14s %9s\n", "stack", "time", "allocs", "bytes", "vs floor")
    base = nothing
    for (name, f, _) in STACKS
        try
            s = minimum(@be gen_draws() f)
            base === nothing && (base = s.time)
            @printf("%-16s %12s %12d %14s %8.1fx\n",
                    name, fmt_time(s.time), s.allocs, fmt_bytes(s.bytes), s.time / base)
        catch e
            @printf("%-16s  errored: %s\n", name, sprint(showerror, e))
        end
    end
end

fmt_time(t) = t < 1e-6 ? @sprintf("%.1f ns", t*1e9) :
              t < 1e-3 ? @sprintf("%.1f μs", t*1e6) :
              t < 1.0  ? @sprintf("%.2f ms", t*1e3) : @sprintf("%.2f s", t)
fmt_bytes(b) = b < 1024 ? "$(b) B" :
               b < 1024^2 ? @sprintf("%.1f KiB", b/1024) : @sprintf("%.2f MiB", b/1024^2)

check_equivalence()
run_benchmarks()
