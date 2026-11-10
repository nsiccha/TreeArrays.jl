# ===================== cells -> tree: one value per draw, converted once (decision 0jwqiy5) =====================
# A model's return values arrive as an array with one CELL per draw (FlexiChains'
# `returned(model, chain)` is a `(iter x chain)` DimMatrix), and a cell is whatever the model
# returned: numbers, (nested) NamedTuples, arrays, arrays of arrays. `_fromcells` transposes that
# array-of-structs ONCE into a proper tree over the outer (draw) dims -- the struct-of-arrays
# shape the 0.2.0 refusal (`_StructsBacking`, decision 1c2lsko) tells users to build by hand:
#
#   number per cell              -> a dense leaf over the outer dims (zero-copy when the cells
#                                   already are a numeric array)
#   NamedTuple per cell          -> a record (axis `:field`); a field that is a number per cell is
#                                   stored raw over the outer dims, like FlexiChains' `:param` record
#   equal-size numeric arrays    -> one dense leaf over (outer..., inner...), one copy
#   arrays of arrays / of records-> a RAGGED node over the array's own positions, whose children are
#                                   converted the same way (participants with different trial counts)
#
# Inner axes are named by the caller's `axes =` spec, which mirrors the value's structure: a
# NamedTuple at a record, a Tuple of Symbols at an array (its first `ndims` names name that level,
# the rest pass down to the children). Unnamed inner axes are `:elem` (one dense level) or
# `:elem_1`, `:elem_2`, ... -- as in the FlexiChains ingest. Inner axes are labelled `1:n`, so
# `align =` can match them across participants. Anything else refuses by name, with the path.

_pathstr(path) = isempty(path) ? "the cells" : "`" * join(map(string, path), ".") * "`"

function _fromcells(cells::AbstractArray, outer::Tuple, spec, path::Tuple, off::Int = 0)
    isempty(cells) && throw(ArgumentError("TreeArrays: cannot convert " * _pathstr(path) *
        ": there are no cells (no draws) to read its shape from."))
    _fromcells(first(cells), cells, outer, spec, path, off)
end

_allnumbers(c) = eltype(c) <: Number || all(x -> x isa Number, c)

function _fromcells(::Number, cells, outer, spec, path, off)
    _allnumbers(cells) || _mixedcells(cells, path, "a number")
    _nospec(spec, path, "a number per draw")
    TreeData(eltype(cells) <: Number ? cells : map(identity, cells), outer...)
end

function _fromcells(c1::NamedTuple, cells, outer, spec, path, off)
    ks = keys(c1)
    for c in cells
        c isa NamedTuple && keys(c) == ks || _mixedcells(cells, path, "a NamedTuple with fields " * string(ks))
    end
    spec === nothing || spec isa NamedTuple || throw(ArgumentError("TreeArrays: `axes` names " *
        _pathstr(path) * " with a " * string(nameof(typeof(spec))) * ", but it is a record (fields " *
        string(ks) * "): name its fields' axes with a NamedTuple, e.g. `(" * string(first(ks)) * " = (:a, :b),)`."))
    if spec isa NamedTuple
        extra = filter(k -> !(k in ks), keys(spec))
        isempty(extra) || throw(ArgumentError("TreeArrays: `axes` names fields " * string(extra) * " that " *
            _pathstr(path) * " does not have (fields: " * string(ks) * ")."))
    end
    fields = map(k -> _fromcells(map(c -> c[k], cells), outer, spec === nothing ? nothing : get(spec, k, nothing),
                                 (path..., k), off), ks)
    raw = map(f -> f isa TreeArray && !(parent(f) isa AbstractArray{<:TreeData}) &&
                   isequal(TreeArrays.dims(f), outer) ? parent(f) : f, fields)
    TreeData(:field => NamedTuple{ks}(raw), outer...)
end

function _fromcells(c1::AbstractArray, cells, outer, spec, path, off)
    sz = size(c1)
    for c in cells
        c isa AbstractArray || _mixedcells(cells, path, "an array")
        size(c) == sz || throw(ArgumentError("TreeArrays: cannot convert " * _pathstr(path) * ": its array " *
            "size differs between draws (" * string(sz) * " vs " * string(size(c)) * "). A value's shape must " *
            "not vary from draw to draw."))
    end
    if all(_allnumbers, cells)                                   # one dense leaf over (outer..., inner...)
        names = _axnames(spec, length(sz), path, off, true)
        T = mapreduce(eltype, promote_type, cells)
        out = Array{T}(undef, size(cells)..., sz...)
        inner = ntuple(_ -> Colon(), length(sz))
        for I in CartesianIndices(cells)
            copyto!(view(out, Tuple(I)..., inner...), cells[I])
        end
        TreeData(out, outer..., map((n, s) -> TreeDim(n, 1:s), names, sz)...)
    else                                                          # ragged: one child per position
        names, rest = _axnames(spec, length(sz), path, off, false)
        kids = map(i -> _fromcells(map(c -> c[i], cells), outer, rest, (path..., Tuple(i)...), off + length(sz)),
                   CartesianIndices(c1))
        TreeData(kids, map((n, s) -> TreeDim(n, 1:s), names, sz)...)
    end
end

_fromcells(c1, cells, outer, spec, path, off) = throw(ArgumentError("TreeArrays: cannot convert " *
    _pathstr(path) * ": a cell is a `" * string(typeof(c1)) * "`; expected numbers, NamedTuples or arrays of them."))

_mixedcells(cells, path, want) = throw(ArgumentError("TreeArrays: cannot convert " * _pathstr(path) *
    ": its cells are not all " * want * " (the first is a `" * string(typeof(first(cells))) * "`). A value's " *
    "type and fields must not vary from draw to draw."))

_nospec(::Nothing, path, what) = nothing
_nospec(spec, path, what) = throw(ArgumentError("TreeArrays: `axes` names axes for " * _pathstr(path) *
    ", but it is " * what * " and has no inner axes."))

# Names for an array level with `n` axes. Dense: the spec must name exactly `n`. Ragged: the
# first `n` name this level and the rest pass down (`nothing` once used up).
function _axnames(spec, n, path, off, dense::Bool)
    if spec === nothing
        names = dense && off == 0 && n == 1 ? (:elem,) : ntuple(i -> Symbol(:elem_, off + i), n)
        return dense ? names : (names, nothing)
    end
    spec isa Tuple && all(s -> s isa Symbol, spec) || throw(ArgumentError("TreeArrays: `axes` names " *
        _pathstr(path) * " with a " * string(nameof(typeof(spec))) * ", but it is an array: name its axes " *
        "with a Tuple of Symbols, e.g. `(:participant, :trial)`."))
    if dense
        length(spec) == n || throw(ArgumentError("TreeArrays: `axes` gives " * string(length(spec)) * " name(s) " *
            string(spec) * " for " * _pathstr(path) * ", which has " * string(n) * " inner axis(es)."))
        return spec
    end
    length(spec) >= n || throw(ArgumentError("TreeArrays: `axes` gives " * string(length(spec)) * " name(s) " *
        string(spec) * " for " * _pathstr(path) * ", which nests at least " * string(n + 1) * " levels of axes."))
    (spec[1:n], length(spec) == n ? nothing : spec[n+1:end])
end
