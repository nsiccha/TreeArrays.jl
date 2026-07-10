# ===================== rich markdown display =====================
# The `MIME"text/markdown"` half of `html.jl`. An HTMXObjects `?plain` / `?markdown`
# request (or `Accept: text/markdown`) renders the value through this MIME (htmxo-use
# §8), and `?plain` is how agents read routes -- so this is a normal-use path, not a
# nicety.
#
# ⚠ WHY A METHOD HERE IS BOTH NECESSARY AND SUFFICIENT. Unlike the HTML rep, HTMX's
# markdown rep does NOT guard on `showable`: it carries a catch-all
# `show(io, ::MIME"text/markdown", val) = print(io, string(val))` (HTMX.jl:246) that
# matches anything. So a `TreeData` was rendering as `string(td)` -- and `string` sets
# no `:limit`. A method on `::TreeData` is more specific than that `val` catch-all, so
# dispatch prefers it: no HTMXObjects dependency, exactly like the HTML method.
# (Mechanism read out of HTMX.jl by `HTMXObjects:consult.tfgq0l`, 2026-07-10.)
#
# The catch-all was only half the story: `print_tree` set no `:limit` either, so
# `string(td)` / `@info td` dumped the whole backing array (916_987 chars for a
# 1000x100 leaf). That root is fixed in `show.jl`; this file is the rich rep, not a
# way around it.
#
# Bounded exactly as `html.jl` is -- same constants, same helpers, same two rules:
# never densify, and MARK every elision (an unmarked truncation reads as the whole
# value, which is the "valid-looking value" the reduction invariants forbid).

# A `|` closes a cell in a pipe table and a newline closes the ROW, so both must be
# neutralized -- but the two sources of cell text need different treatment:
#
#   `_md`     for SHOW-derived text (coord previews, Symbol names). `show` already
#             escaped it to a single line and its backslashes are meaningful escape
#             sequences, so touching them would double-escape (`a\nb` -> `a\\nb`).
#             Only the pipe is left to handle.
#
#   `_mdcell` for PRINT-derived text (table cell VALUES). `print` emits a String raw,
#             so a coordinate label like "a\nb" landed a literal newline mid-row and
#             split the table. Backslash goes first (single-pass `replace`, so nothing
#             is escaped twice), otherwise a value ending in `\` would escape the cell
#             delimiter we emit right after it.
_md(s::AbstractString) = replace(s, '|' => "\\|")
_md(x) = _md(string(x))

_mdcell(v) = replace(sprint(print, v; context = :compact => true),
    '\\' => "\\\\", '|' => "\\|", '\n' => "\\n", '\r' => "\\r")

function _mddimtable(io::IO, X::TreeData)
    print(io, "**", sprint(print_type, typeof(X)), "**\n\n")
    print(io, "| dim | kind | coords |\n| --- | --- | --- |\n")
    for d in TreeArrays.dims(X)
        print(io, "| `", _md(name(d)), "` | ", _kindlabel(X, d),
            " | `", _md(_coordpreview(meta(d).values)), "` |\n")
    end
    print(io, "\n")
end

function Base.show(io::IO, m::MIME"text/markdown", X::TreeData)
    _mddimtable(io, X)
    _mdvalues(io, m, X)
end

# array- / tuple- / scalar-backed leaf: the bounded plain-text dump, fenced.
_mdvalues(io::IO, ::MIME"text/markdown", X::TreeData) =
    print(io, "```\n", _plaindump(parent(X)), "\n```\n")

function _mdvalues(io::IO, m::MIME"text/markdown", X::TreeNamedTuple)
    for (k, v) in pairs(parent(X))
        print(io, "**", _md(k), "**\n\n")
        _mdchild(io, m, v)
    end
end

function _mdvalues(io::IO, m::MIME"text/markdown", X::TreeRaggedArray)
    p = parent(X)
    n = length(p)
    shown = min(n, _MAX_LEAVES)
    for i in 1:shown
        print(io, "**[", i, "]**\n\n")
        show(io, m, p[i])
    end
    shown < n && print(io, "… ", n - shown, " more leaves\n")
end

_mdchild(io::IO, m::MIME"text/markdown", v::TreeData) = show(io, m, v)   # knows its own dims
_mdchild(io::IO, ::MIME"text/markdown", v) = print(io, "```\n", _plaindump(v), "\n```\n")

# ---- TreeTable: a pipe table off the lazy melt, same `_MAX_ROWS` cap as the HTML rep.
function Base.show(io::IO, ::MIME"text/markdown", tt::TreeTable)
    cols = Tables.columns(tt)
    nms = keys(cols)
    n = isempty(nms) ? 0 : length(first(cols))
    shown = min(n, _MAX_ROWS)
    print(io, "**TreeTable** — ", n, " row", n == 1 ? "" : "s")
    shown < n && print(io, " (showing the first ", shown, ")")
    print(io, "\n\n|")
    for nm in nms
        print(io, " `", _md(nm), "` |")
    end
    print(io, "\n|", repeat(" --- |", length(nms)), "\n")
    for i in 1:shown
        print(io, "|")
        for nm in nms
            print(io, " ", _mdcell(cols[nm][i]), " |")
        end
        print(io, "\n")
    end
end
