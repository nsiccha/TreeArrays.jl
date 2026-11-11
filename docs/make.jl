using Documenter, DocumenterVitepress

using TreeArrays
using AlgebraOfVega: vega_vendor_dir

run(`python3 $(joinpath(@__DIR__, "lint.py"))`)

makedocs(;
    modules  = [TreeArrays],
    authors  = "Nikolas Siccha",
    repo     = "https://github.com/nsiccha/TreeArrays.jl",
    sitename = "TreeArrays.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo      = "github.com/nsiccha/TreeArrays.jl",
        devurl    = "dev",
        devbranch = "main",
        build_vitepress = false,
    ),
    pages = [
        "Start" => ["TreeArrays" => "index.md"],
        "Guides" => [
            "Named axes and records" => "getting-started.md",
            "Reductions" => "reductions.md",
            "Ragged data" => "ragged.md",
            "Streaming and caches" => "streaming.md",
            "Interoperability" => "tables.md",
        ],
        "Examples" => ["Concentration trajectories" => "trajectories.md"],
        "Reference" => ["API" => "api.md"],
    ],
    checkdocs = :exports,
    warnonly  = false,
)

# Serve AlgebraOfVega's supported linked runtime from build-only assets.
assets = joinpath(@__DIR__, "build", ".documenter", "public", "assets", "aov")
mkpath(dirname(assets))
cp(vega_vendor_dir(), assets; force=true)
cp(joinpath(@__DIR__, "build", ".documenter", "trajectory-figure.html"),
   joinpath(dirname(assets), "trajectory-figure.html"); force=true)
DocumenterVitepress.build_docs(joinpath(@__DIR__, "build"))

# Root redirect: /TreeArrays.jl/ -> /TreeArrays.jl/dev/
let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head><meta http-equiv="refresh" content="0; url=dev/"></head>
    <body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(;
    repo         = "github.com/nsiccha/TreeArrays.jl",
    devbranch    = "main",
    push_preview = true,
)
