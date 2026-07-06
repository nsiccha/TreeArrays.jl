# Plot-tail of the draws->plots benchmark: the "x2" plot layers of the user's
# 4x2 matrix. Two things are measured:
#
#  (A) plot-build ISOLATED — "reduced ribbon table -> plot object" for each layer,
#      from a materialized wide NamedTuple (backend-independent for a materialized
#      table, so measured once):
#        AoV  -> a Vega-Lite spec dict via `to_vegalite` (no Makie; browser renders).
#        AoG  -> a drawn Makie figure via `draw` (CairoMakie, headless).
#
#  (B) the TA-NATIVE draws->AoV cell — TreeArrays reduces (population mean over
#      :subject, then a ONE-PASS 7-level ribbon quantile over :draw) and feeds AoV.
#      NB (per TreeArrays, authoritative): TA's Tables bridge is architecturally
#      LONG (coord columns + one :value column) — it CANNOT emit wide q025…q975
#      columns, and AoV's lineribbon(bands=…) needs WIDE lo/hi columns. So a small
#      base-Julia pivot (long -> wide) is required; there is no lazy no-materialize
#      path for a ribbon. The one-pass quantile keeps TA's 7-levels-in-one-sort
#      advantage (looping 7 scalar quantiles would understate it). treearrays-use
#      §9 overpromised the wide mapping; TreeArrays is fixing the skill + a durable
#      wide-emit seam.
#
# Run:  julia --project=benchmark/plot benchmark/plot/plot_benchmarks.jl

using AlgebraOfVega                      # data, mapping, visual, lineribbon, to_vegalite
using AlgebraOfGraphics: draw            # AoG render entry point
using CairoMakie                         # Band, Lines + headless Cairo backend
using TreeArrays                         # TreeData, TreeDim, mean, quantile (bare = TA idiom)
using Chairmarks, Printf
import Statistics
import Tables

const N_DRAW = 1000
const N_SUBJ = 179
const N_TIME = 50
const BANDS      = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)   # posterior ribbon levels
const BANDSV     = collect(BANDS)
const BANDNAMES  = (:q025, :q10, :q25, :q50, :q75, :q90, :q975)

gen_draws() = randn(N_DRAW, N_SUBJ, N_TIME)   # [draw x subject x time]

# A materialized (time x band) ribbon summary — stands for ANY backend's reduced
# output (built once, NOT timed; the isolated plot-build consumes it).
function reduced_summary_nt()
    mm = dropdims(Statistics.mean(gen_draws(); dims = 2); dims = 2)   # [draw x time]
    q  = Matrix{Float64}(undef, N_TIME, length(BANDSV))
    for t in 1:N_TIME
        q[t, :] .= Statistics.quantile(@view(mm[:, t]), BANDSV)
    end
    merge((time = collect(1:N_TIME),),
          NamedTuple{BANDNAMES}(ntuple(i -> q[:, i], length(BANDNAMES))))
end
const SUMMARY = reduced_summary_nt()

# --- AoV: reduced wide table -> Vega-Lite spec dict ---------------------------
plot_aov(nt) = to_vegalite(
    data(nt) * mapping(:time, :q50) *
    lineribbon(bands = [:q025 => :q975, :q10 => :q90, :q25 => :q75]))

# --- AoG: reduced wide table -> drawn Makie figure (CairoMakie) ---------------
# AoG has no `lineribbon`; compose nested Bands + a median Lines layer.
plot_aog(nt) = draw(
    data(nt) * mapping(:time, :q025, :q975) * visual(Band; alpha = 0.2) +
    data(nt) * mapping(:time, :q10,  :q90)  * visual(Band; alpha = 0.3) +
    data(nt) * mapping(:time, :q25,  :q75)  * visual(Band; alpha = 0.4) +
    data(nt) * mapping(:time, :q50)         * visual(Lines))

# --- TA-native: raw draws -> reduced TreeData -> pivot to wide -> AoV ----------
# Base-Julia pivot of TA's LONG (:time, :band, :value) melt into a wide NamedTuple.
function ta_long_to_wide(long)
    c  = Tables.columns(long)
    ts, bs, vs = c.time, c.band, c.value
    times = sort(unique(ts))
    tix   = Dict(t => i for (i, t) in enumerate(times))
    W     = ntuple(_ -> Vector{Float64}(undef, length(times)), length(BANDNAMES))
    for k in eachindex(ts)
        bi = findfirst(≈(bs[k]), BANDSV)
        W[bi][tix[ts[k]]] = vs[k]
    end
    merge((time = times,), NamedTuple{BANDNAMES}(W))
end

function ta_native_aov(x)
    X    = TreeData(x, :draw, :subject, :time => 1:N_TIME)
    Y    = mean(X; dims = :subject)                       # population mean -> draw × time
    long = quantile(Y, TreeDim(:band, BANDS); dims = :draw)  # ONE-PASS 7-level ribbon -> time × band (long)
    plot_aov(ta_long_to_wide(long))                       # pivot to wide, then AoV spec
end

fmt_time(t) = t < 1e-6 ? @sprintf("%.1f ns", t*1e9) :
              t < 1e-3 ? @sprintf("%.1f μs", t*1e6) :
              t < 1.0  ? @sprintf("%.2f ms", t*1e3) : @sprintf("%.2f s", t)
fmt_bytes(b) = b < 1024 ? "$(b) B" :
               b < 1024^2 ? @sprintf("%.1f KiB", b/1024) : @sprintf("%.2f MiB", b/1024^2)

row(name, s) = @printf("%-30s %12s %12d %14s\n", name, fmt_time(s.time), s.allocs, fmt_bytes(s.bytes))

function main()
    # sanity: every path produces a plot object
    println("AoV spec:   ", typeof(plot_aov(SUMMARY)))
    println("AoG figure: ", typeof(plot_aog(SUMMARY)))
    println("TA-native AoV spec: ", typeof(ta_native_aov(gen_draws())))

    println("\n(A) plot-build from a materialized ribbon ($N_TIME x $(length(BANDSV)) bands)  (@be, min)\n")
    @printf("%-30s %12s %12s %14s\n", "plot layer", "time", "allocs", "bytes")
    row("AoV (to_vegalite spec)", minimum(@be SUMMARY plot_aov))
    row("AoG (Makie draw)",       minimum(@be SUMMARY plot_aog))

    println("\n(B) TA-native draws→AoV  (mean:subject → 1-pass ribbon:draw → pivot → spec)  (@be, min)\n")
    @printf("%-30s %12s %12s %14s\n", "cell", "time", "allocs", "bytes")
    row("TreeArrays → AoV (native)", minimum(@be gen_draws() ta_native_aov))
end

main()
