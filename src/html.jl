# ===================== rich HTML display =====================
# `showable(MIME"text/html"(), x)` is exactly what HTMX.jl's element-builder loop
# checks before falling back to `print(io, child)`. So defining these two methods
# is ALL it takes for a `TreeData`/`TreeTable` dropped straight into an `h.div(...)`
# to render rich -- no HTMXObjects dependency, no package extension, no consumer
# opt-in (user steer, 2026-07-10: "'more automatic' rich display would be the
# feature to reach for"). `HTMX.jl:202` is the branch, and `m` is pinned by its
# enclosing `show(::IO, m::MIME"text/html", ::Node)` signature, so the call is
# exactly `show(io, MIME"text/html"(), child)`.
#
# TA owns the methods rather than HTMXO/AoV because `MIME` and `show` are Base and
# the dependency arrow only ever points HTMXO -> TA: a method here costs TA nothing
# and reaches every consumer at once (Pluto and Jupyter light up for free).
#
# ⚠ HTML REP ONLY. HTMX's markdown rep takes a different path with a
# `show(io, ::MIME"text/markdown", val) = print(io, string(val))` CATCH-ALL and no
# `showable` guard, so an HTMXO `?plain` request silently renders `string(x)` --
# `print_tree`, which sets no `:limit` and dumped 917 KB for a 1000x100 leaf where
# this method emits 919 chars. Fixing that needs a `MIME"text/markdown"` method
# here (todo `5ou6j6`), not a change to the HTML one.
#
# BOUNDED BY CONSTRUCTION -- a display method must never densify the tree (eager
# compute / lazy assembly, decision 1uzarfr):
#   * the dim table is computed from the dims alone, never from the values;
#   * array-backed leaves go through Base's `:limit => true` plain-text show, which
#     only touches the array corners;
#   * TUPLE-backed leaves and tuple coords are elided by US, up front -- Base honours
#     `:limit` for arrays but NOT for `Tuple` (see `show.jl`, where the shared bounded
#     helpers `_clamp` / `_showcoords` / `_coordpreview` / `_plaindump` live);
#   * a ragged container previews `_MAX_LEAVES` leaves and then says how many it
#     skipped -- never a silent truncation;
#   * `TreeTable` pulls at most `_MAX_ROWS` rows out of the lazy melt columns;
#   * `_clamp` is the final backstop on any single `<pre>` dump.

# HTMX splices a `showable` child's output into the document RAW -- `HTMX.jl:202`
# takes `show(io, MIME"text/html"(), child)` and never escapes it (that same branch
# is why `h.code()(src)` passes text through unescaped). So the EMITTER owns
# escaping: a `<` in an axis label would otherwise corrupt the document, and is
# XSS-adjacent once a coordinate carries user data. Confirmed against HTMX.jl by
# `HTMXObjects:consult.tfgq0l`, 2026-07-10.
#
# Escapes the full `& < > " '` set even though every call site below emits element
# TEXT (where the quote forms are inert), so `_esc` is a CONTEXT-FREE escaper: a
# later edit that drops an escaped value into an attribute stays safe by
# construction rather than by remembering this comment.
# `replace` with several pairs is single-pass, so `&` -> `&amp;` is not re-escaped.
_esc(s::AbstractString) = replace(s,
    '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;", '\'' => "&#39;")
_esc(x) = _esc(string(x))

# `_clamp` runs BEFORE `_esc` at every call site below, so an HTML entity can never be
# cut in half.
_predump(io::IO, v) = print(io, "<pre>", _esc(_plaindump(v)), "</pre>")

# `_dimkind` (tables.jl) classifies :axis/:ghost/:fixed; the record axis is the one
# dim that class can't see, since being the field-enumerating axis is a property of
# the CONTAINER (`outer_dim`), not of the dim.
_isrecdim(X::TreeData, d::TreeDim) =
    haskey(meta(X), :outer_dim) && name(d) === name(outerdim(X))
_kindlabel(X::TreeData, d::TreeDim) = _isrecdim(X, d) ? :record : _dimkind(d)

function _dimtable(io::IO, X::TreeData)
    print(io, "<table><caption><code>")
    print_type(io, typeof(X))
    print(io, "</code></caption><thead><tr><th>dim</th><th>kind</th><th>coords</th></tr></thead><tbody>")
    for d in TreeArrays.dims(X)
        print(io, "<tr><td><code>", _esc(name(d)), "</code></td><td>", _kindlabel(X, d),
            "</td><td><code>", _esc(_coordpreview(meta(d).values)), "</code></td></tr>")
    end
    print(io, "</tbody></table>")
end

function Base.show(io::IO, m::MIME"text/html", X::TreeData)
    print(io, "<div class=\"treedata\">")
    _dimtable(io, X)
    _htmlvalues(io, m, X)
    print(io, "</div>")
end

# array- / tuple- / scalar-backed leaf: a bounded plain-text dump of the backing.
_htmlvalues(io::IO, ::MIME"text/html", X::TreeData) = _predump(io, parent(X))

# a record shows one collapsible section per field, recursing so each field renders
# with its OWN dim table -- this is what makes a nested posterior legible.
function _htmlvalues(io::IO, m::MIME"text/html", X::TreeNamedTuple)
    for (k, v) in pairs(parent(X))
        print(io, "<details><summary><code>", _esc(k), "</code></summary>")
        _htmlchild(io, m, v)
        print(io, "</details>")
    end
end

function _htmlvalues(io::IO, m::MIME"text/html", X::TreeRaggedArray)
    p = parent(X)
    n = length(p)
    shown = min(n, _MAX_LEAVES)
    for i in 1:shown
        print(io, "<details><summary>[", i, "]</summary>")
        show(io, m, p[i])
        print(io, "</details>")
    end
    shown < n && print(io, "<p>… ", n - shown, " more leaves</p>")
end

_htmlchild(io::IO, m::MIME"text/html", v::TreeData) = show(io, m, v)   # knows its own dims
_htmlchild(io::IO, ::MIME"text/html", v) = _predump(io, v)             # raw field / `missing` sentinel

# ---- TreeTable: the tabular view renders as an actual table -------------------
# Columns stay the lazy melt views (`ConstColumn`/`AxisColumn`/`ValueColumn`); only
# the `_MAX_ROWS` previewed cells are ever read. Orientation is whatever the view
# was built with: a `wide=` pivot (7465185) renders its levels as columns, exactly
# as `Tables.columns` hands them over. Nothing here special-cases a shape -- a
# shape `Tables.columns` rejects (a ragged tree) throws straight through this
# method too, because display must not be the one place a known gap renders as a
# silently-wrong table (htmxo-use §3.5: let errors bubble).
function Base.show(io::IO, ::MIME"text/html", tt::TreeTable)
    cols = Tables.columns(tt)
    nms = keys(cols)
    n = isempty(nms) ? 0 : length(first(cols))
    shown = min(n, _MAX_ROWS)
    print(io, "<table><caption>TreeTable — ", n, " row", n == 1 ? "" : "s")
    shown < n && print(io, " (showing the first ", shown, ")")
    print(io, "</caption><thead><tr>")
    for nm in nms
        print(io, "<th><code>", _esc(nm), "</code></th>")
    end
    print(io, "</tr></thead><tbody>")
    for i in 1:shown
        print(io, "<tr>")
        for nm in nms
            print(io, "<td>", _esc(sprint(print, cols[nm][i]; context = :compact => true)), "</td>")
        end
        print(io, "</tr>")
    end
    print(io, "</tbody></table>")
end
