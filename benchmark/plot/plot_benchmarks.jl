# Plot-tail of the draws->plots benchmark: the "x2" plot layers of the user's
# 4x2 matrix. Times "reduced ribbon summary -> plot object" for both layers:
#
#   AoV  -> a Vega-Lite spec dict via `to_vegalite` (no Makie; browser renders).
#   AoG  -> a drawn Makie figure via `draw` (CairoMakie, headless).
#
# The reduced (time x band) ribbon summary is output-EQUAL across all data
# backends (see ../benchmarks.jl), so the materialized-table plot-build is
# backend-independent — measured once here on a wide NamedTuple. The TA-NATIVE
# lazy path (TreeData -> AoV via the view-struct Tables bridge, no materialize)
# is the one backend-dependent cell — added once TreeArrays confirms the exact
# `data(reduced_treedata) * mapping * lineribbon` incantation.
#
# Run:  julia --project=benchmark/plot benchmark/plot/plot_benchmarks.jl

using AlgebraOfVega                      # data, mapping, visual, lineribbon, to_vegalite (AoG algebra re-exported)
using AlgebraOfGraphics: draw            # AoG render entry point
using CairoMakie                         # Band, Lines + headless Cairo backend
using Chairmarks, Printf
import Statistics

const N_DRAW = 1000
const N_SUBJ = 179
const N_TIME = 50
const BANDSV     = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
const BANDNAMES  = (:q025, :q10, :q25, :q50, :q75, :q90, :q975)

# The materialized (time x band) ribbon summary every data backend produces
# (built once, NOT timed — the plot layers consume it).
function reduced_summary_nt()
    x  = randn(N_DRAW, N_SUBJ, N_TIME)
    mm = dropdims(Statistics.mean(x; dims = 2); dims = 2)     # [draw x time] population mean
    q  = Matrix{Float64}(undef, N_TIME, length(BANDSV))
    for t in 1:N_TIME
        q[t, :] .= Statistics.quantile(@view(mm[:, t]), BANDSV)
    end
    merge((time = collect(1:N_TIME),),
          NamedTuple{BANDNAMES}(ntuple(i -> q[:, i], length(BANDNAMES))))
end
const SUMMARY = reduced_summary_nt()

# --- AoV: reduced table -> Vega-Lite spec dict --------------------------------
plot_aov(nt) = to_vegalite(
    data(nt) * mapping(:time, :q50) *
    lineribbon(bands = [:q025 => :q975, :q10 => :q90, :q25 => :q75]))

# --- AoG: reduced table -> drawn Makie figure (CairoMakie) --------------------
# AoG has no `lineribbon`; compose nested Bands + a median Lines layer.
plot_aog(nt) = draw(
    data(nt) * mapping(:time, :q025, :q975) * visual(Band; alpha = 0.2) +
    data(nt) * mapping(:time, :q10,  :q90)  * visual(Band; alpha = 0.3) +
    data(nt) * mapping(:time, :q25,  :q75)  * visual(Band; alpha = 0.4) +
    data(nt) * mapping(:time, :q50)         * visual(Lines))

function main()
    # sanity: both produce a plot object
    sv = plot_aov(SUMMARY)
    println("AoV spec: ", typeof(sv), sv isa AbstractDict ? " ($(length(sv)) top-level keys)" : "")
    fg = plot_aog(SUMMARY)
    println("AoG figure: ", typeof(fg))

    println("\nplot-build from the reduced ribbon summary ($N_TIME x $(length(BANDSV)) bands)  (@be, min sample)\n")
    @printf("%-22s %12s %12s %14s\n", "plot layer", "time", "allocs", "bytes")
    for (name, f) in [("AoV (to_vegalite spec)", plot_aov), ("AoG (Makie draw)", plot_aog)]
        try
            s = minimum(@be SUMMARY f)
            @printf("%-22s %12s %12d %14s\n", name, fmt_time(s.time), s.allocs, fmt_bytes(s.bytes))
        catch e
            @printf("%-22s  errored: %s\n", name, sprint(showerror, e))
        end
    end
end

fmt_time(t) = t < 1e-6 ? @sprintf("%.1f ns", t*1e9) :
              t < 1e-3 ? @sprintf("%.1f μs", t*1e6) :
              t < 1.0  ? @sprintf("%.2f ms", t*1e3) : @sprintf("%.2f s", t)
fmt_bytes(b) = b < 1024 ? "$(b) B" :
               b < 1024^2 ? @sprintf("%.1f KiB", b/1024) : @sprintf("%.2f MiB", b/1024^2)

main()
