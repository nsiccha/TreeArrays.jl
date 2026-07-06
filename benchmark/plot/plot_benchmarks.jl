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
#      :subject, then a ONE-PASS multi-level ribbon quantile over :draw) and feeds
#      AoV with ZERO hand-rolling. As of TreeArrays `c57b847`, `quantile(Y, p::NamedTuple;
#      dims)` returns a RECORD whose fields melt to WIDE columns natively (q025…q975) —
#      exactly the shape AoV's `lineribbon(bands=…)` consumes — so the reduced TreeData
#      feeds `data(...)` directly: no pivot, no materialize. (Earlier this cell needed a
#      base-Julia long->wide pivot because the Tables bridge was long-only; the native
#      wide-emit removed it — the user requirement "TA+AoV just works".)
#
# Run:  julia --project=benchmark/plot benchmark/plot/plot_benchmarks.jl

using AlgebraOfVega                      # data, mapping, visual, lineribbon, to_vegalite
using AlgebraOfGraphics: draw            # AoG render entry point
using CairoMakie                         # Band, Lines + headless Cairo backend
using TreeArrays                         # TreeData, mean, quantile (bare = TA idiom)
using Chairmarks, Printf
import Statistics
import Tables

const N_DRAW = 1000
const N_SUBJ = 179
const N_TIME = 50
const BANDSV     = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
const BANDNAMES  = (:q025, :q10, :q25, :q50, :q75, :q90, :q975)
# level=>name NamedTuple: TA's native wide-emit — each field becomes a melted column.
const WIDEP      = (q025 = 0.025, q10 = 0.1, q25 = 0.25, q50 = 0.5, q75 = 0.75, q90 = 0.9, q975 = 0.975)

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

# --- TA-native: raw draws -> reduced TreeData (native WIDE record) -> AoV ------
# Zero hand-rolling: `quantile(Y, WIDEP; dims=:draw)` emits q025…q975 as melted
# columns (TA c57b847), fed straight to AoV — no long->wide pivot.
function ta_native_aov(x)
    X = TreeData(x, :draw, :subject, :time => 1:N_TIME)
    Y = mean(X; dims = :subject)                 # population mean -> draw × time
    plot_aov(quantile(Y, WIDEP; dims = :draw))   # 1-pass multi-level ribbon -> wide record -> AoV spec
end

# Output-equivalence: TA's native wide-emit must match a base-Julia type-7 ref
# (row-aligned by the melted :time column, so ordering-agnostic).
function ta_native_maxabsdiff()
    x    = gen_draws()
    mm   = dropdims(Statistics.mean(x; dims = 2); dims = 2)        # draw × time
    wide = quantile(mean(TreeData(x, :draw, :subject, :time => 1:N_TIME); dims = :subject),
                    WIDEP; dims = :draw)
    c    = Tables.columns(wide)
    tcol = collect(c.time)
    Δ = 0.0
    for (i, nm) in enumerate(BANDNAMES)
        col = collect(getproperty(c, nm))
        for k in eachindex(tcol)
            ref = Statistics.quantile(@view(mm[:, Int(tcol[k])]), BANDSV[i])
            Δ = max(Δ, abs(col[k] - ref))
        end
    end
    Δ
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

    Δ = ta_native_maxabsdiff()
    @printf("\nTA-native wide-emit output-equivalence vs base-Julia type-7: max |Δ| = %.2e  %s\n",
            Δ, Δ == 0 ? "✓ EXACT" : Δ < 1e-10 ? "✓" : "✗ MISMATCH")

    println("\n(A) plot-build from a materialized ribbon ($N_TIME x $(length(BANDSV)) bands)  (@be, min)\n")
    @printf("%-30s %12s %12s %14s\n", "plot layer", "time", "allocs", "bytes")
    row("AoV (to_vegalite spec)", minimum(@be SUMMARY plot_aov))
    row("AoG (Makie draw)",       minimum(@be SUMMARY plot_aog))

    println("\n(B) TA-native draws→AoV  (mean:subject → 1-pass wide-record ribbon:draw → spec, NO pivot)  (@be, min)\n")
    @printf("%-30s %12s %12s %14s\n", "cell", "time", "allocs", "bytes")
    row("TreeArrays → AoV (native)", minimum(@be gen_draws() ta_native_aov))
end

main()
