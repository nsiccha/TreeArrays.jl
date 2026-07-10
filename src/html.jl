# ===================== rich HTML display =====================
# `showable(MIME"text/html"(), x)` is exactly what HTMX.jl's element-builder loop
# checks before falling back to `print(io, child)` (htmxo-use §8a). So defining
# these two methods is ALL it takes for a `TreeData`/`TreeTable` dropped straight
# into an `h.div(...)` to render rich -- no HTMXObjects dependency, no package
# extension, no consumer opt-in (user steer, 2026-07-10: "'more automatic' rich
# display would be the feature to reach for").
#
# TA owns the methods rather than HTMXO/AoV because `MIME` and `show` are Base and
# the dependency arrow only ever points HTMXO -> TA: a method here costs TA nothing
# and reaches every consumer at once.
#
# BOUNDED BY CONSTRUCTION -- a display method must never densify the tree (eager
# compute / lazy assembly, decision 1uzarfr):
#   * the dim table is computed from the dims alone, never from the values;
#   * leaf values go through Base's `:limit => true` plain-text show, which only
#     touches the array corners;
#   * a ragged container previews `_MAX_LEAVES` leaves and then says how many it
#     skipped -- never a silent truncation;
#   * `TreeTable` pulls at most `_MAX_ROWS` rows out of the lazy melt columns.

const _MAX_ROWS = 20     # TreeTable preview rows
const _MAX_LEAVES = 3    # ragged-container preview leaves
const _MAX_COORDS = 60   # chars of a dim's coordinate preview

_esc(s::AbstractString) = replace(s, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
_esc(x) = _esc(string(x))

# compact one-liner for a dim's coordinate values (`1:1000`, `(:a, :b)`, `missing`, …)
function _coordpreview(v)
    s = sprint(show, v; context = (:limit => true, :compact => true))
    length(s) > _MAX_COORDS ? first(s, _MAX_COORDS) * "…" : s
end

# Base's plain-text show already truncates under `:limit`; the small displaysize
# keeps a 1000x100 backing array to a corner preview rather than a megabyte of HTML.
_predump(io::IO, v) = print(io, "<pre>", _esc(
    sprint(show, MIME"text/plain"(), v; context = (:limit => true, :displaysize => (12, 80)))
), "</pre>")

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
# the `_MAX_ROWS` previewed cells are ever read. A `wide=` pivot that isn't built
# yet throws out of `Tables.columns` here exactly as it does everywhere else --
# a display method must not be the one place a known gap renders as a silent long
# table (htmxo-use §3.5: let errors bubble).
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
