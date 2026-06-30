module TreeArraysWeb

using HTMXObjects
import Markdown

# REQUIREMENTS.md lives at the repo root (web/src/../../REQUIREMENTS.md).
# Read fresh per request so doc edits show up live — no restart needed.
const REQUIREMENTS_PATH = normpath(joinpath(@__DIR__, "..", "..", "REQUIREMENTS.md"))
const OPTION_A_PATH = joinpath(@__DIR__, "option_a_dd.jl")
const OPTION_B_PATH = joinpath(@__DIR__, "option_b_mock.jl")

# Comparison artifacts (NOT src/ — the build-on-DD decision is exactly what they
# inform). Order matters: BenchKit before the options that `using ..BenchKit`.
include("shared_data.jl")
include("bench.jl")
include("option_a_dd.jl")
include("option_b_mock.jl")

# ── Data layer (DO): synthetic input + the two computed/benchmarked runs,
# memoized once so the per-step timing is computed a single time, not per
# request (do-use §10 — no access-and-discard).
@dynamicstruct struct AppData
    data  = SharedData.synthetic_data()
    run_a = OptionA.run(data)
    run_b = OptionB.run(data)
end
const APPDATA = AppData()

# ── Formatting helpers ──────────────────────────────────────────────────────
fmt_seconds(s) =
    s < 1e-6 ? string(round(s * 1e9; digits=1), " ns") :
    s < 1e-3 ? string(round(s * 1e6; digits=2), " µs") :
    s < 1.0  ? string(round(s * 1e3; digits=2), " ms") :
               string(round(s; digits=2), " s")
fmt_bytes(b) =
    b < 1024    ? string(b, " B") :
    b < 1024^2  ? string(round(b / 1024; digits=1), " KiB") :
                  string(round(b / 1024^2; digits=2), " MiB")
fmt(x) = string(round(x; digits=4))

# Code lines = non-blank, non-comment lines (rough ergonomics proxy).
count_code_lines(src) =
    count(l -> (t = strip(l); !isempty(t) && !startswith(t, "#")), split(src, '\n'))

# ── Per-option rendering ────────────────────────────────────────────────────
steps_table(steps) = h.table(
    h.thead(h.tr(h.th("Step"), h.th("Time"), h.th("Allocated"), h.th("What happens"))),
    h.tbody((h.tr(
        h.td(s.name), h.td(fmt_seconds(s.seconds)), h.td(fmt_bytes(s.bytes)), h.td(s.note),
    ) for s in steps)...),
)

function output_summary(out)
    waste = out.ragged_padded_n / out.ragged_n_values
    h.article(
        h.header(h.strong("Computed output — real numbers from the synthetic data")),
        h.ul(
            h.li("Variance ratio (zero / in_sample): random-effect columns (",
                 string(out.n_re), ") mean ≈ ", h.strong(fmt(out.ratio_re_mean)),
                 "; all other columns (", string(out.n_other), ") mean ≈ ",
                 h.strong(fmt(out.ratio_other_mean)),
                 " — zeroing the RE columns drops their variance to 0, leaving the rest untouched."),
            h.li("Cross-subject trough quantiles (10 / 50 / 90 %): ",
                 join(fmt.(out.cross_quantiles), ", ")),
            h.li("Ragged measurement storage: ", string(out.ragged_n_values),
                 " values; the padded subject×time rectangle is ",
                 string(out.ragged_padded_n), " cells (", fmt(waste), "× the data)."),
        ),
        h.p("First 3 subjects' per-series stats (trough, peak, baseline, Δtrough, Δpeak):"),
        h.ul((h.li(string(round.(values(s); digits=3))) for s in out.stats_sample)...),
    )
end

function source_block(path)
    src = read(path, String)
    h.details(
        h.summary("Implementation source — ", basename(path), " (",
                  string(count_code_lines(src)), " code lines)"),
        h.pre(h.code(escape_html(src))),
    )
end

render_run(run, title, intro, path) = h.section(
    h.h2(title),
    h.p(intro),
    output_summary(run.outputs),
    h.h3("Per-step timing & allocation"),
    steps_table(run.steps),
    h.h3("Source"),
    source_block(path),
)

# ── Benchmarks: side-by-side + ergonomics + verdict ─────────────────────────
alloc_ratio(a_bytes, b_bytes) =
    b_bytes == 0 ? "—" : string(round(a_bytes / max(b_bytes, 1); digits=1), "×")

function benchmarks_table(a, b)
    rows = map(zip(a.steps, b.steps)) do (sa, sb)
        h.tr(
            h.td(sa.name),
            h.td(fmt_seconds(sa.seconds)), h.td(fmt_seconds(sb.seconds)),
            h.td(fmt_bytes(sa.bytes)), h.td(fmt_bytes(sb.bytes)),
            h.td(h.strong(alloc_ratio(sa.bytes, sb.bytes))),
        )
    end
    h.table(
        h.thead(h.tr(
            h.th("Step"),
            h.th("A time (DD)"), h.th("B time (mock)"),
            h.th("A alloc (DD)"), h.th("B alloc (mock)"), h.th("A/B alloc"),
        )),
        h.tbody(rows...),
    )
end

function ergonomics_section(a, b)
    la = count_code_lines(read(OPTION_A_PATH, String))
    lb = count_code_lines(read(OPTION_B_PATH, String))
    h.article(
        h.header(h.strong("Ergonomics")),
        h.ul(
            h.li("Code lines — Option A (DD): ", string(la), "; Option B (mock): ", string(lb),
                 ". DD is terser for the rectangular slice (named-dim reductions, ",
                 h.code("var(; dims=:draw)"), " for free); the mock pays lines for its CSR + structural-stack types."),
            h.li(h.strong("Assembly friction (DD): "),
                 "no structural/streamed assembly — ", h.code("cat"),
                 " eagerly materializes the full combined block. There is no way to express ",
                 "\"describe the cross-product, never allocate it\"."),
            h.li(h.strong("Ragged friction (DD): "),
                 "fundamentally rectangular — the per-subject series must be padded to a ",
                 h.code("Union{Missing,Float64}"), " rectangle, and the per-group kernel drops to ",
                 h.code("eachrow(parent(...))"), " + ", h.code("skipmissing"),
                 ". The mock's ", h.code("mapslices_ragged"),
                 " hands ", h.code("f"), " an exact contiguous view — the same ", h.code("f"),
                 " a rectangular input would use (invariant §7)."),
        ),
    )
end

function verdict_section(a, b)
    sa = a.steps[3]; sb = b.steps[3]              # the stack step
    total_a = sum(s.bytes for s in a.steps)
    total_b = sum(s.bytes for s in b.steps)
    waste = a.outputs.ragged_padded_n / a.outputs.ragged_n_values
    tiny = minimum(s.seconds for s in a.steps) < 1e-6
    h.article(
        h.header(h.strong("Verdict — build on DimensionalData, or standalone?")),
        h.p("The allocation profile is decisive (per the design's central claim), so the ",
            "verdict leans on bytes, not wall-time."),
        h.ul(
            h.li(h.strong("Assembly: "), "the ", h.code("stack"), " step allocates ",
                 fmt_bytes(sa.bytes), " under DD (it ", h.code("cat"),
                 "s into the materialized combined block) vs ", fmt_bytes(sb.bytes),
                 " for the mock's structural stack (", alloc_ratio(sa.bytes, sb.bytes),
                 " more). This is the \"never allocate the combined matrix\" differentiator, and it is real."),
            h.li(h.strong("Ragged: "), "DD pads to ", fmt(waste),
                 "× the data and forces skipmissing; the mock's CSR is exact. Total run allocation: ",
                 fmt_bytes(total_a), " (DD) vs ", fmt_bytes(total_b), " (mock)."),
            h.li(h.strong("Wall-time: "),
                 tiny ? "several steps are sub-microsecond — at this size timing is noise; the allocation gap and its O(combined) vs O(block) scaling carry the conclusion." :
                        "see the table; the allocation gap is the more robust signal and scales with data size."),
        ),
        h.p(h.strong("Recommendation: "),
            "Reuse DD's load-bearing GOOD ideas — dims-as-types (compile-time axis resolution) and ",
            "the ", h.code("dims=:name"), " reduction ergonomics — but NOT its eager-materialize core ",
            "or its rectangular-only storage. Those two are exactly where TreeArrays' value is: ",
            "structural (non-allocating) assembly that streams block-by-block, and native ragged (CSR) ",
            "axes. Both are things DD cannot provide without the workarounds shown above. ",
            "Net: a standalone design (optionally borrowing DD's dim-type machinery, not building ON DD), ",
            "with structural assembly + CSR ragged as the non-negotiable core."),
    )
end

render_benchmarks(a, b) = h.section(
    h.h2("Benchmarks — Option A (DimensionalData) vs Option B (mock TreeArrays)"),
    h.p("Per-step wall-time and allocated bytes for the same slice, same seeded input. ",
        "The differentiators (no combined-matrix allocation; no padding waste) show up in ",
        h.strong("bytes"), ", which is why allocation drives the verdict."),
    benchmarks_table(a, b),
    ergonomics_section(a, b),
    verdict_section(a, b),
)

# Sidebar nav. Paths are root-relative; `nav_sidebar(…; prefix=__prefix__)`
# prepends the per-request mount prefix so links resolve both locally and
# under the `/p/TreeArrays/` proxy mount.
# Full, descriptive labels — never truncated; the user reads these.
nav_items() = [
    "Overview"                   => "/",
    "Requirements"               => "/requirements",
    "Option A — DimensionalData" => "/option_a",
    "Option B — mock TreeArrays" => "/option_b",
    "Benchmarks"                 => "/benchmarks",
]

@htmx struct AppContext
    __appdata__ = APPDATA

    # Per-request mount prefix from the reverse proxy's `X-Forwarded-Prefix`
    # header: empty for direct local/LAN access (links emit at `/…`), and
    # `/p/TreeArrays` under the KB reverse-proxy (links emit at `/p/TreeArrays/…`).
    # The proxy strips the prefix before forwarding, since routes register at root.
    __prefix__ = isnothing(__req__) ? "" : HTTP.header(__req__, "X-Forwarded-Prefix", "")

    @get index() = h.section(
        h.hgroup(
            h.h1("TreeArrays.jl"),
            h.p("Comparison & benchmark harness — DimensionalData vs. a mock TreeArrays."),
        ),
        h.p("TreeArrays.jl is a postprocessing / summarization layer for Bayesian posterior draws and PKPD model predictions: named, hierarchical, possibly-ragged arrays over flat backing storage, with eager compute / lazy assembly and streaming reductions."),
        h.p(
            "The load-bearing open question is whether to build it on top of DimensionalData or standalone. To decide empirically, a coherent runnable subset of the ",
            h.code("usage/main.jl"),
            " example is implemented two ways and compared on ergonomics and performance:",
        ),
        h.ul(
            h.li(h.strong("Requirements"), " — the distilled spec (", h.code("REQUIREMENTS.md"), ")."),
            h.li(h.strong("Option A — DimensionalData"), " — the example implemented with DimensionalData.jl."),
            h.li(h.strong("Option B — mock TreeArrays"), " — the same example with a mock standalone TreeArrays."),
            h.li(h.strong("Benchmarks"), " — side-by-side timing, allocation, ergonomics, and the build-on-DD verdict."),
        ),
        h.p("The slice exercises both deliberate differentiators end-to-end: structural (non-allocating) assembly and ragged per-subject axes. Use the sidebar to navigate."),
    )

    # REQUIREMENTS.md. Returning a `Markdown.MD` lets the response pipeline
    # serve HTML to browsers and the raw markdown on `?plain` (both via
    # `showable`) — no request-mode branching in the body.
    @get requirements() = Markdown.parse(read(REQUIREMENTS_PATH, String))

    @get option_a() = render_run(
        __appdata__.run_a, "Option A — DimensionalData",
        "The slice implemented with DimensionalData.jl — eager, materializing assembly (cat); ragged data padded to a missing-rectangle.",
        OPTION_A_PATH,
    )

    @get option_b() = render_run(
        __appdata__.run_b, "Option B — mock TreeArrays",
        "The same slice with a mock standalone TreeArrays — structural (non-allocating) assembly; native CSR ragged axis.",
        OPTION_B_PATH,
    )

    @get benchmarks() = render_benchmarks(__appdata__.run_a, __appdata__.run_b)

    # Prefix-aware sidebar — built as a property so `__prefix__` is in scope;
    # referenced by `__page__` below (mirrors the Heizung pattern).
    sidebar = nav_sidebar(nav_items(); prefix=__prefix__)

    __page__ = content -> htmx(
        app_layout(sidebar, content);
        pico_version="2",
    )
end

function __init__()
    route!(AppContext())
end

end # module TreeArraysWeb
