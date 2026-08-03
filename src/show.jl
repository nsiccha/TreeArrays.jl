function Base.show(io::IO, T::Type{<:TreeDim})
    if get(io, :compact, false)
        print(io, name(T))
    else
        invoke(show, Tuple{IO, Type}, io, T)  # default
    end
end
function Base.show(io::IO, T::Type{<:TreeData})
    if get(io, :compact, false)
        print_type(io, T)
        print_dims(io, T)
    else
        invoke(show, Tuple{IO, Type}, io, T)  # default
    end
end

print_type(io::IO, ::Type{<:TreeData}) = print(io, "TreeData")
print_type(io::IO, ::Type{<:TreeNamedTuple}) = print(io, "TreeNamedTuple")
print_type(io::IO, ::Type{<:TreeRaggedArray}) = print(io, "TreeRaggedArray")
print_type(io::IO, ::Type{<:TreeArray}) = print(io, "TreeArray")
print_dims(io::IO, T::Type{<:TreeData}) = begin
    M = fieldtype(T, :meta)
    # A NON-concrete TreeData type carries no dim list to print. Julia reaches here on its
    # own: a container whose leaves differ in their dim TYPES (ragged `:time` coords given as
    # Tuples of different lengths, say) has a widened eltype, and `show`ing that container
    # prints the eltype as a typeinfo prefix. Say "unknown" rather than throw from inside
    # `show` -- a display method must never be the thing that errors.
    isconcretetype(M) || return print(io, "(…)")
    ds = fieldtypes(fieldtype(M, :dims))
    ds = :outer_dim in fieldnames(M) ? (ds..., fieldtype(M, :outer_dim)) : ds
    print(io, "("); join(io, ds, ", "); print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", x::TreeData) = show(io, x)

Base.show(io::IO, X::TreeData)  = print_tree(io, X)
print_tree(io::IO, X::TreeData) = begin
    compact = get(io, :compact, false)
    cio = IOContext(io, :compact => true)
    # @info get(io, :typeinfo, nothing)
    get(io, :typeinfo, Any) == typeof(X) ||  print(cio, typeof(X))
    compact && return print_values(cio, X)
    print(io, ":\n")
    print_dims(io, X)
    print_values(io, X)
end
print_dims(io::IO, X::TreeData) = begin
    print(io, "----------------\n")
    for dim in dims(X)
        print(io, dim, "\n")
    end
    haskey(meta(X), :outer_dim) && print(io, outerdim(X), "\n")
    print(io, "----------------")
end
# ===================== bounded preview helpers (shared by every rep) =====================
# Plain text, HTML and markdown all need the SAME two things: cap how much of a value
# is rendered, and MARK every elision (a preview that reads as the whole value is the
# "valid-looking value" the reduction invariants forbid). They live here, with `show`,
# rather than in `html.jl` -- markdown already had to reach across for them, and the
# plain rep needs them most of all (see the `:limit` holes below).
const _MAX_ROWS = 20          # TreeTable preview rows
const _MAX_LEAVES = 3         # ragged-container preview leaves
const _MAX_COORDS = 60        # chars of a dim's coordinate preview
const _MAX_TUPLE_ELEMS = 8    # Tuple elements rendered before eliding (`show` ignores `:limit`)
const _MAX_DUMP_CHARS = 2000  # hard cap on any single value dump

# clamp on CHARACTERS, before any escaping -- escaping last means an escape sequence
# can never be cut in half.
_clamp(s::AbstractString, n::Int, mark::AbstractString) =
    length(s) > n ? first(s, n) * mark : s

_rawshow(v) = sprint(show, v; context = (:limit => true, :compact => true))

# THE `:limit` ASYMMETRY. Base honours `:limit => true` for AbstractArrays -- `show_vector`
# / `print_matrix` render only the corners -- but NOT for `Tuple`, whose `show` renders
# every element regardless. So a Tuple must be elided by US, up front, or a 4000-name
# record axis builds a ~128 MB string just to be cut to `_MAX_COORDS` chars.
_showcoords(v) = _rawshow(v)                       # arrays/ranges: `show` honours `:limit`
function _showcoords(v::Tuple)                     # tuples: it does not -- elide up front
    n = length(v)
    n <= _MAX_TUPLE_ELEMS && return _rawshow(v)
    head = join((_rawshow(v[i]) for i in 1:_MAX_TUPLE_ELEMS), ", ")
    string("(", head, ", … ", n - _MAX_TUPLE_ELEMS, " more)")
end
_coordpreview(v) = _clamp(_showcoords(v), _MAX_COORDS, "…")

# the small displaysize keeps a 1000x100 backing array to a corner preview rather than a
# megabyte of output; `_clamp` backstops any type Base declines to limit.
_plaindump(v) = _clamp(sprint(show, MIME"text/plain"(), v;
    context = (:limit => true, :displaysize => (12, 80))), _MAX_DUMP_CHARS, "\n… (truncated)")
function _plaindump(v::Tuple)
    n = length(v)
    n <= _MAX_TUPLE_ELEMS && return _clamp(sprint(show, MIME"text/plain"(), v;
        context = (:limit => true, :displaysize => (12, 80))), _MAX_DUMP_CHARS, "\n… (truncated)")
    head = join((_rawshow(v[i]) for i in 1:_MAX_TUPLE_ELEMS), ", ")
    _clamp(string("(", head, ", …)  # ", n, " elements"), _MAX_DUMP_CHARS, "\n… (truncated)")
end

# ===================== the plain-text rep, bounded =====================
# `print` sets no `:limit`, so the backing array was dumped WHOLE by anything that
# did not happen to set it: `string(td)`, a `@info` line, and -- the reason this was
# found -- HTMX's markdown rep, whose `show(io, ::MIME"text/markdown", val)` catch-all
# is `print(io, string(val))`. Measured 916_987 chars for a 1000x100 leaf; the same
# value under `:limit` is 337. The REPL sets `:limit` itself, which is why display
# always looked fine.
#
# So default to limited. Base marks its own elisions (`…`), so this never presents a
# truncated array as if it were whole -- and an explicit `IOContext(io, :limit=>false)`
# still gets the full dump, so the escape hatch survives.
_limitctx(io::IO) = IOContext(io, :compact => true,
    :limit => get(io, :limit, true), :displaysize => get(io, :displaysize, (12, 80)))

# `:limit` alone does NOT bound a Tuple-backed leaf (the asymmetry above): a 3000-element
# `TreeTuple` still dumped 22_950 chars. Elide it ourselves, exactly as the coord preview
# does. `_limitctx` still governs array leaves, where Base does the right thing.
_printvalue(io::IO, v) = print(_limitctx(io), v)
_printvalue(io::IO, v::Tuple) = print(io, _plaindump(v))

print_values(io::IO, X::TreeData) = begin
    get(io, :compact, false) || print(io, "\n")
    _printvalue(io, parent(X))
end
print_values(io::IO, X::TreeNamedTuple) = if get(io, :compact, false)
    print(_limitctx(io), parent(X))
else
    print(io, "\n")
    io = _limitctx(io)
    for (k, v) in pairs(parent(X))
        print(io, k, ": ", v, "\n")   # `v` is a TreeData: recurses, `:limit` propagates
    end
end

# A dim's coordinates were the OTHER unbounded hole on this path: `print(cio, values)`
# set no `:limit` at all, so `print_dims` dumped every coordinate -- 59_164 chars for a
# 4000-name axis (31_161 even for a Vector, since `:limit` was never set). `_coordpreview`
# is what the HTML and markdown dim tables already used; the plain rep just never got it.
Base.show(io::IO, D::TreeDim) = print(io, name(D), ": ", _coordpreview(meta(D).values))
