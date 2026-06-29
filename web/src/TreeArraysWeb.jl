module TreeArraysWeb

using HTMXObjects
import Markdown

# REQUIREMENTS.md lives at the repo root (web/src/../../REQUIREMENTS.md).
# Read fresh per request so doc edits show up live — no restart needed.
const REQUIREMENTS_PATH = normpath(joinpath(@__DIR__, "..", "..", "REQUIREMENTS.md"))

# Sidebar nav. The app is root-mounted, so the paths are absolute.
# Full, descriptive labels — never truncated; the user reads these.
nav_items() = [
    "Overview"                   => "/",
    "Requirements"               => "/requirements",
    "Option A — DimensionalData" => "/option_a",
    "Option B — mock TreeArrays" => "/option_b",
    "Benchmarks"                 => "/benchmarks",
]

# Placeholder for the parts that land in later runs. Semantic elements
# only (Pico styles them); `coming` lists what slots in here next.
_stub(title, intro, coming) = h.section(
    h.h2(title),
    h.p(intro),
    h.article(
        h.header(h.strong("Lands in a later run")),
        h.ul((h.li(item) for item in coming)...),
    ),
)

@htmx struct AppContext

    @get index() = h.section(
        h.hgroup(
            h.h1("TreeArrays.jl"),
            h.p("Comparison & benchmark harness — DimensionalData vs. a mock TreeArrays."),
        ),
        h.p("TreeArrays.jl is a postprocessing / summarization layer for Bayesian posterior draws and PKPD model predictions: named, hierarchical, possibly-ragged arrays over flat backing storage, with eager compute / lazy assembly and streaming reductions."),
        h.p(
            "The load-bearing open question is whether to build it on top of DimensionalData or standalone. To decide empirically, the ",
            h.code("usage/main.jl"),
            " example will be implemented two ways and compared on ergonomics and performance:",
        ),
        h.ul(
            h.li(h.strong("Requirements"), " — the distilled spec (", h.code("REQUIREMENTS.md"), ")."),
            h.li(h.strong("Option A — DimensionalData"), " — the example implemented with DimensionalData.jl."),
            h.li(h.strong("Option B — mock TreeArrays"), " — the example implemented with a mock TreeArrays."),
            h.li(h.strong("Benchmarks"), " — side-by-side ergonomics and timing."),
        ),
        h.p("This page is the scaffolding; the two option implementations and the benchmark table land in later runs. Use the sidebar to navigate."),
    )

    # REQUIREMENTS.md. Returning a `Markdown.MD` lets the response pipeline
    # serve HTML to browsers and the raw markdown on `?plain` (both via
    # `showable`) — no request-mode branching in the body.
    @get requirements() = Markdown.parse(read(REQUIREMENTS_PATH, String))

    @get option_a() = _stub(
        "Option A — DimensionalData",
        "The usage/main.jl example implemented with DimensionalData.jl.",
        ["the implementation code (rendered)",
         "its computed output (summaries / predictions)",
         "per-step timing for the benchmark comparison"],
    )

    @get option_b() = _stub(
        "Option B — mock TreeArrays",
        "The usage/main.jl example implemented with a mock TreeArrays.",
        ["the implementation code (rendered)",
         "its computed output (summaries / predictions)",
         "per-step timing for the benchmark comparison"],
    )

    @get benchmarks() = _stub(
        "Benchmarks",
        "Side-by-side comparison of the two options — ergonomics and performance.",
        ["a per-step timing table (Option A vs Option B)",
         "ergonomics notes / line counts",
         "the build-on-DimensionalData vs standalone verdict"],
    )

    __page__ = content -> htmx(
        app_layout(nav_sidebar(nav_items()), content);
        pico_version="2",
    )
end

function __init__()
    route!(AppContext())
end

end # module TreeArraysWeb
