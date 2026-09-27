module TreeArraysFlexiChainsExt

using TreeArrays
using TreeArrays: meta, name, _isaxis
using FlexiChains: FlexiChains, FlexiChain, FlexiSummary, Parameter, Extra

# ===================== ingest: FlexiChain -> TreeData =====================
# A FlexiChain maps Parameter/Extra keys to (niters x nchains) matrices. The TA
# home is a record: one field per key over the inner axes `:draw` (FC iters) +
# `:chain`, with the record axis named `:param`.
#
# Three facts shape this code (all probed against FlexiChains v0.6.40):
# - `chain[k]` COPIES (DimensionalData keyword-getindex materializes), and no
#   public raw-matrix accessor exists -- so ingest copies once per key at
#   construction. That copy is a construction cost, not a reduction, so the
#   no-eager-restructuring invariant is untouched.
# - `Symbol(::Parameter)` preserves VarName optics textually (`y[1]`), so the
#   flat-Symbol field rule works for VN chains with no extra mangling.
# - array-valued cells are stacked by the loop below (not `stack=true`) so
#   ragged/exotic cells fail with TA's message, not DimensionalData's.

"""
    TreeData(chain::FlexiChain; extras=false)

Ingest a FlexiChains posterior chain as a [`TreeData`](@ref) record: one field
per chain key over the inner axes `:draw` (chain iterations) + `:chain`, with
the record axis named `:param`.

```julia
X = TreeData(chain)                      # parameters only
X = TreeData(chain; extras=true)         # also ingest `Extra` keys (e.g. `:lp`)
mean(X; dims=:draw)                      # ordinary TA reductions from here
```

- Field names are `Symbol(key)` -- `VarName` optics survive textually (`y[1]`,
  `y[2]` stay distinct). A `Parameter`/`Extra` pair collapsing to one name
  throws (mirroring FlexiChains' own `Symbol`-index ambiguity rule).
- Scalar (`Number`, or `missing`-or-`Number`) keys become `(ndraw, nchain)`
  matrix fields. Anything else scalar-ish (strings, heterogeneous cells)
  throws by name -- nothing is silently dropped.
- Array-valued keys (every cell an array) nest as a sub-`TreeData`: the cells
  stack into one dense `(draw, chain, elem...)` leaf with unlabelled `:elem*`
  axes. Ragged or mixed-eltype cells throw by name.
- `:draw`/`:chain` coords come from the chain's index lookups, so
  `discard_initial`/thinning offsets survive the crossing.
- Cost: one copy per key at construction (`chain[k]` materializes; there is no
  public zero-copy accessor). Reductions on the result never copy.
"""
function TreeArrays.TreeData(chain::FlexiChain; extras::Bool=false)
    keys_in = extras ? collect(keys(chain)) : Parameter.(FlexiChains.parameters(chain))
    names = _keysyms(keys_in)
    drawdim = TreeDim(:draw, collect(FlexiChains.iter_indices(chain)))
    chaindim = TreeDim(:chain, collect(FlexiChains.chain_indices(chain)))
    fields = map(k -> _chainfield(chain, k, drawdim, chaindim), keys_in)
    TreeData(:param => NamedTuple{tuple(names...)}(tuple(fields...)), drawdim, chaindim)
end

function _keysyms(ks)
    names = map(_keysym, ks)
    seen = Dict{Symbol,Any}()
    for (k, nm) in zip(ks, names)
        if haskey(seen, nm)
            throw(ArgumentError(
                "TreeArrays: FlexiChains keys `$(seen[nm])` and `$k` both flatten to record field " *
                "`:$nm` -- ambiguous, ingesting neither (this mirrors FlexiChains' own `Symbol`-index " *
                "ambiguity rule: index with the full key or rename upstream)."))
        end
        seen[nm] = k
    end
    names
end

function _keysym(k)
    try
        return Symbol(k)
    catch e
        e isa MethodError || rethrow()
        throw(ArgumentError(
            "TreeArrays: cannot ingest FlexiChains key `$k`: its name has no `Symbol` spelling " *
            "(`Symbol` names and `VarName`s are supported)."))
    end
end

function _chainfield(chain::FlexiChain, k, drawdim::TreeDim, chaindim::TreeDim)
    mat = parent(chain[k])::AbstractMatrix
    T = eltype(mat)
    if T <: AbstractArray
        return _nest_matrix_of_arrays(mat, (drawdim, chaindim), "FlexiChains key `$k`")
    elseif !(nonmissingtype(T) <: Number)
        throw(ArgumentError(
            "TreeArrays: cannot ingest FlexiChains key `$k` with value type `$T`: only `Number` " *
            "(or `missing`-or-`Number`) and uniformly-shaped `AbstractArray` values are supported -- " *
            "nothing is silently dropped."))
    end
    mat
end

# stack a matrix-of-arrays into one dense (base..., elem...) leaf. `what` names
# the key in every error. Element axes are unlabelled positions (`:elem`, or
# `:elem_1`, `:elem_2`, ... for multi-dim elements): the chain carries no
# element labels, so none are invented.
function _nest_matrix_of_arrays(mat::AbstractMatrix, basedims::Tuple{Vararg{TreeDim}}, what::String)
    isempty(mat) && throw(ArgumentError(
        "TreeArrays: cannot ingest array-valued $what: it holds zero cells, so the element shape " *
        "cannot be inferred."))
    c0 = first(mat)
    sz0, ET = size(c0), eltype(c0)
    for (i, cell) in enumerate(mat)
        size(cell) == sz0 || throw(ArgumentError(
            "TreeArrays: cannot ingest array-valued $what: cell $i has size $(size(cell)) where cell 1 " *
            "has size $sz0 -- ragged element shapes cannot nest; split the key upstream."))
        eltype(cell) == ET || throw(ArgumentError(
            "TreeArrays: cannot ingest array-valued $what: cell $i has eltype $(eltype(cell)) where cell 1 " *
            "has eltype $ET -- mixed element types cannot nest; split the key upstream."))
    end
    ni, nc = size(mat)
    cube = Array{ET}(undef, ni, nc, sz0...)
    inner = ntuple(_ -> Colon(), length(sz0))
    for j in 1:nc, i in 1:ni
        @views cube[i, j, inner...] .= mat[i, j]
    end
    elemdims = length(sz0) == 1 ? (TreeDim(:elem),) :
        ntuple(i -> TreeDim(Symbol(:elem_, i)), length(sz0))
    TreeData(cube, basedims..., elemdims...)
end

# ===================== emit: TreeData -> FlexiChain =====================
# The inverse crossing, through FlexiChains' own 3D-array constructor
# (`iters x chains x params` + key spec, which copies into per-key matrices).
# Rectangular numeric trees only; everything else is refused by name, mirroring
# `TreeActualArray`'s contract. Axis names are strict (`:draw`/`:chain`/`:param`,
# the shapes ingest produces and TA uses natively): a `:subject` axis is not a
# chain axis, and no kwarg reinterprets it as one.

"""
    FlexiChain(X::TreeData)

Emit a rectangular numeric tree as a `FlexiChain{Symbol}`.

```julia
chain = FlexiChain(X)   # dense 3-d leaf or homogeneous record in, chain out
```

Accepted shapes (anything else throws by name):

- a dense leaf whose axis dims are exactly `:draw`, `:chain`, `:param` in any
  order (permuted as needed); the `:param` labels become the keys;
- a flat record `TreeData(:param => (; ...), :draw, :chain)` (inner order either
  way) with `(ndraw, nchain)` matrix fields; nested sub-`TreeData` fields
  unstack back to array-valued keys.

Semantics:

- numeric fields promote to one common eltype (like FlexiChains' own `Long`
  format); non-numeric fields throw. Scalar fields throw -- a scalar is not
  per-draw data and broadcasting it would invent draws.
- integer `:draw`/`:chain` coords pass through as the chain's index lookups
  (offsets round-trip); unlabelled axes default to `1:n`; non-integer coords
  throw.
- fixed (non-axis) dims are allowed but do not survive -- a chain has no home
  for them.
"""
FlexiChains.FlexiChain(X::TreeData) = _emit_chain(X)

_emit_chain(X::TreeRaggedArray) = throw(ArgumentError(
    "TreeArrays: cannot emit a ragged (array-of-trees) `TreeData` as a `FlexiChain`: a chain needs " *
    "one (iters x chains) matrix per key -- feed a dense leaf " *
    "(`TreeData(arr3d, :draw, :chain, :param)`) or a homogeneous record " *
    "(`TreeData(:param => (; ...), :draw, :chain)`)."))
_emit_chain(X::TreeTuple) = throw(ArgumentError(
    "TreeArrays: cannot emit a `Tuple`-backed record as a `FlexiChain`: name the record axis with a " *
    "NamedTuple record (`TreeData(:param => (; ...), :draw, :chain)`)."))

function _emit_chain(X::TreeArray)
    axnames = [name(d) for d in TreeArrays.dims(X) if _isaxis(d)]
    _require_draw_chain_param(axnames, "dense leaf")
    _requirenumeric(eltype(parent(X)), "dense leaf")
    pd = findfirst(==(:draw), axnames)
    pc = findfirst(==(:chain), axnames)
    cube = (pd, pc) == (1, 2) ? Array(parent(X)) : permutedims(parent(X), [pd, pc, 6 - pd - pc])
    names = _param_symbols(collect(coords(X, :param)), "dense leaf `:param` axis")
    ni, nc = size(cube, 1), size(cube, 2)
    FlexiChain{Symbol}(cube, tuple(Parameter.(names)...);
        iter_indices=_index_or_default(X, :draw, ni),
        chain_indices=_index_or_default(X, :chain, nc))
end

function _emit_chain(X::TreeNamedTuple)
    haskey(meta(X), :outer_dim) || throw(ArgumentError(
        "TreeArrays: cannot emit this record as a `FlexiChain`: it has no named record axis -- build it " *
        "as `TreeData(:param => (; fields...), :draw, :chain)` so the fields form an axis."))
    rec = name(outerdim(X))
    rec == :param || throw(ArgumentError(
        "TreeArrays: cannot emit this record as a `FlexiChain`: its record axis is `:$rec`, and a chain " *
        "needs `:param`."))
    inner = [name(d) for d in TreeArrays.dims(X) if name(d) != :param && _isaxis(d)]
    sort(inner) == [:chain, :draw] || throw(ArgumentError(
        "TreeArrays: cannot emit this record as a `FlexiChain`: its inner axes are $inner, and a chain " *
        "needs exactly `:draw` + `:chain`."))
    p = parent(X)
    isempty(p) && throw(ArgumentError(
        "TreeArrays: cannot emit an empty record as a `FlexiChain`: a chain needs at least one key."))
    swap = inner[1] == :chain
    names = collect(keys(p))
    # (ni, nc) come from the FIELDS, not the dims: an unlabelled dim reports
    # `length == 1` (the `_coords` wrap), so dims cannot size the data. Labelled
    # dims are cross-checked against (ni, nc) in `_index_or_default`.
    ni, nc = _probe_draw_chain(p, names, swap)
    cols, spec = _emit_columns(p, names, ni, nc, swap)
    FlexiChain{Symbol}(cols, spec;
        iter_indices=_index_or_default(X, :draw, ni),
        chain_indices=_index_or_default(X, :chain, nc))
end

function _require_draw_chain_param(axnames, what)
    sort(axnames) == [:chain, :draw, :param] || throw(ArgumentError(
        "TreeArrays: cannot emit this $what as a `FlexiChain`: its axis dims are $axnames, and a chain " *
        "needs exactly `:draw` + `:chain` + `:param`."))
    nothing
end

function _requirenumeric(T::Type, what)
    nonmissingtype(T) <: Number || throw(ArgumentError(
        "TreeArrays: cannot emit $what with eltype `$T` as a `FlexiChain`: only `Number` " *
        "(or `missing`-or-`Number`) values survive the crossing."))
    nothing
end

function _dim_by_name(X::TreeData, nm::Symbol)
    for d in TreeArrays.dims(X)
        name(d) == nm && return d
    end
    throw(ArgumentError("TreeArrays: no dim named `:$nm` on this node -- it carries $(map(name, TreeArrays.dims(X)))."))
end

function _param_symbols(vals::AbstractVector, what)
    names = map(v -> Symbol(string(v)), vals)
    seen, dups = Set{Symbol}(), Symbol[]
    for nm in names
        nm in seen ? push!(dups, nm) : push!(seen, nm)
    end
    isempty(dups) || throw(ArgumentError(
        "TreeArrays: cannot emit $what as `FlexiChain` keys: labels $(unique(dups)) collide after " *
        "`Symbol` conversion."))
    names
end

function _index_or_default(X::TreeData, nm::Symbol, n::Int)
    v = meta(_dim_by_name(X, nm)).values
    v isa Missing && return collect(1:n)
    if v isa Union{Tuple,AbstractVector,AbstractRange} && all(x -> x isa Integer, v)
        length(v) == n || throw(ArgumentError(
            "TreeArrays: dim `:$nm` carries $(length(v)) labels for an axis of length $n -- refusing to " *
            "guess the alignment."))
        return collect(Int, v)
    end
    throw(ArgumentError(
        "TreeArrays: cannot emit dim `:$nm` as FlexiChains indices: its coordinates are not integers, and " *
        "chain indices must be (relabel with integer positions, or leave the axis unlabelled for the " *
        "default `1:$n`)."))
end

# the first matrix/dense field fixes (ni, nc); every field (including that one)
# is then validated against it in `_emit_block`, so disagreement still throws.
# With no sizable field at all, the first field's own error is the accurate one.
function _probe_draw_chain(p::NamedTuple, names::Vector{Symbol}, swap::Bool)
    for nm in names
        v = getfield(p, nm)
        s = v isa AbstractMatrix ? size(v) :
            (v isa TreeArray && !(v isa TreeRaggedArray) && ndims(parent(v)) >= 2) ? size(parent(v)) : nothing
        s === nothing && continue
        return swap ? (s[2], s[1]) : (s[1], s[2])
    end
    _notperdraw(names[1], getfield(p, names[1]))
end

function _notperdraw(nm::Symbol, v)
    throw(ArgumentError(
        "TreeArrays: cannot emit record field `:$nm` as a `FlexiChain` key: a `$(typeof(v))` is not " *
        "per-draw data (scalar fields would have to be broadcast, which invents draws)."))
end

# column blocks + key spec for the record path, in field order. Scalar fields
# consume one column; nested sub-trees consume `prod(sz)` reshaped column-major,
# exactly inverting `_nest_matrix_of_arrays` (and matching the 3D ctor's own
# `reshape(v, sz)`).
function _emit_columns(p::NamedTuple, names::Vector{Symbol}, ni::Int, nc::Int, swap::Bool)
    blocks = map(names) do nm
        _emit_block(getfield(p, nm), nm, ni, nc, swap)
    end
    P = foldl(promote_type, map(first, blocks); init=Union{})
    ncols = sum(map(b -> b[3], blocks))
    cols = Array{P,3}(undef, ni, nc, ncols)
    spec = map(blocks) do (_, nm, _, _, sz)
        sz == () ? Parameter(nm) : Parameter(nm) => sz
    end
    off = 1
    for (_, _, w, data, _) in blocks
        cols[:, :, off:off+w-1] .= data
        off += w
    end
    cols, tuple(spec...)
end

# one field -> (eltype, name, ncols, data-cube (ni, nc, w), inner-size). Scalar
# matrices normalize orientation to (draw, chain); nested sub-trees must be
# dense leaves over (:draw, :chain, elem...) with matching sizes.
function _emit_block(v::AbstractMatrix, nm::Symbol, ni::Int, nc::Int, swap::Bool)
    size(v) == (swap ? (nc, ni) : (ni, nc)) || throw(ArgumentError(
        "TreeArrays: cannot emit record field `:$nm` as a `FlexiChain` key: it has size $(size(v)) where " *
        "the `:draw`/`:chain` axes need ($ni, $nc)."))
    _requirenumeric(eltype(v), "record field `:$nm`")
    data = swap ? permutedims(v, (2, 1)) : Array(v)
    (eltype(data), nm, 1, reshape(data, ni, nc, 1), ())
end

function _emit_block(v::TreeData, nm::Symbol, ni::Int, nc::Int, swap::Bool)
    # NB `TreeRaggedArray` is a subconstraint of `TreeArray`, so the ragged
    # exclusion is explicit: only a dense leaf unstacks (one nesting level).
    (v isa TreeArray && !(v isa TreeRaggedArray)) || throw(ArgumentError(
        "TreeArrays: cannot emit nested record field `:$nm` as a `FlexiChain` key: only dense-leaf " *
        "sub-trees unstack (one nesting level); got a `$(typeof(v))`."))
    subnames = [name(d) for d in TreeArrays.dims(v) if _isaxis(d)]
    want = swap ? [:chain, :draw] : [:draw, :chain]
    length(subnames) >= 2 && subnames[1:2] == want || throw(ArgumentError(
        "TreeArrays: cannot emit nested record field `:$nm` as a `FlexiChain` key: its leading axes are " *
        "$subnames, and they must start with $want (matching sizes ($ni, $nc), then element axes)."))
    a = parent(v)
    size(a, 1) == (swap ? nc : ni) && size(a, 2) == (swap ? ni : nc) || throw(ArgumentError(
        "TreeArrays: cannot emit nested record field `:$nm` as a `FlexiChain` key: its leading sizes " *
        "($(size(a, 1)), $(size(a, 2))) do not match the `:draw`/`:chain` axes ($ni, $nc)."))
    _requirenumeric(eltype(a), "nested record field `:$nm`")
    sz = tuple(size(a)[3:end]...)
    flat = swap ? permutedims(a, (2, 1, 3:ndims(a)...)) : Array(a)
    (eltype(flat), nm, prod(sz), reshape(flat, ni, nc, prod(sz)), sz)
end

function _emit_block(v, nm::Symbol, ni::Int, nc::Int, swap::Bool)
    _notperdraw(nm, v)
end

# ===================== melt: FlexiSummary -> TreeData =====================
# A summary maps keys to (iters x chains x stats) cubes with collapsed dims
# tracked alongside (`size` reports 0 for them; the lookup is `nothing`). The
# melt keeps the surviving dims as axes: fully-collapsed single-stat summaries
# (all scalars) become a dense `:param` vector, everything else a record whose
# fields ride over the surviving dims in iter->chain->stat order.

"""
    TreeData(st::FlexiSummary)

Melt a FlexiChains summary (`mean`, `summarystats`, `collapse`, ...) as a
[`TreeData`](@ref): one entry per chain key over the dims the summary did not
collapse.

```julia
S = TreeData(summarystats(chain))   # record over the surviving dims + `:stat`
m = TreeData(mean(chain))           # dense `:param` vector (all scalars)
```

- A fully-collapsed single-stat summary (every key a scalar) melts to a dense
  vector over `:param` -- scalar record fields are never produced.
- Otherwise each key becomes a record field over the surviving dims
  (`:draw`/`:chain` for uncollapsed sample dims, `:stat` for uncollapsed
  statistics, coords from the summary's lookups).
- Array-valued stat cells nest exactly like ingest; `missing` stat values are
  preserved as-is (reductions over them fail downstream, loudly, not here).
"""
function TreeArrays.TreeData(st::FlexiSummary)
    ks = collect(keys(st))
    names = _keysyms(ks)
    ni, nc, ns = size(st)
    if ni == 0 && nc == 0 && ns == 0
        vals = map(k -> st[k], ks)
        all(v -> !(v isa AbstractArray), vals) || throw(ArgumentError(
            "TreeArrays: internal inconsistency melting a `FlexiSummary`: `size` reports full collapse " *
            "but some keys still hold arrays."))
        P = foldl(promote_type, map(typeof, vals); init=Union{})
        return TreeData(P[vals...], TreeDim(:param, names))
    end
    surv = TreeDim[]
    FlexiChains.iter_indices(st) !== nothing &&
        push!(surv, TreeDim(:draw, collect(FlexiChains.iter_indices(st))))
    FlexiChains.chain_indices(st) !== nothing &&
        push!(surv, TreeDim(:chain, collect(FlexiChains.chain_indices(st))))
    FlexiChains.stat_indices(st) !== nothing &&
        push!(surv, TreeDim(:stat, collect(FlexiChains.stat_indices(st))))
    basedims = tuple(surv...)
    fields = map(k -> _meltfield(st, k, basedims), ks)
    TreeData(:param => NamedTuple{tuple(names...)}(tuple(fields...)), basedims...)
end

function _meltfield(st::FlexiSummary, k, basedims::Tuple{Vararg{TreeDim}})
    da = st[k]
    da isa AbstractArray || throw(ArgumentError(
        "TreeArrays: internal inconsistency melting `FlexiSummary` key `$k`: expected per-key array data " *
        "but got `$da`."))
    mat = parent(da)
    ndims(mat) == length(basedims) || throw(ArgumentError(
        "TreeArrays: internal inconsistency melting `FlexiSummary` key `$k`: per-key data has rank " *
        "$(ndims(mat)) but $(length(basedims)) dims survived."))
    eltype(mat) <: AbstractArray ?
        _nest_matrix_of_arrays(mat, basedims, "summary key `$k`") : mat
end

end # module
