using Documenter, DocumenterVitepress

using TreeArrays

makedocs(;
    modules  = [TreeArrays],
    authors  = "Nikolas Siccha",
    repo     = "https://github.com/nsiccha/TreeArrays.jl",
    sitename = "TreeArrays.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo      = "github.com/nsiccha/TreeArrays.jl",
        devurl    = "dev",
        devbranch = "main",
    ),
    pages = [
        "Home"                => "index.md",
        "Getting started"     => "getting-started.md",
        "Reductions"          => "reductions.md",
        "Ragged data"         => "ragged.md",
        "Tables & plotting"   => "tables.md",
        "Comparisons"         => "comparisons.md",
        "API"                 => "api.md",
    ],
    checkdocs = :none,
    warnonly  = true,
)

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
