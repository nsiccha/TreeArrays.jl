using TreeArrays, DataFrames, LazyArrays, Statistics


struct TreeDim{N,M<:NamedTuple}
    meta::M
    TreeDim(N, meta::NamedTuple) = new{N,typeof(meta)}(meta)
end
TreeDim(N, values) = TreeDim(N, (;values))
TreeDim(N::Symbol) = TreeDim(N, missing)
TreeDim((N,values)::Pair) = TreeDim(N, values)
name(::TreeDim{N}) where N = N
name(::Type{<:TreeDim{N}}) where N = N
meta(X::TreeDim) = getfield(X, :meta)
sliced(X::TreeDim) = TreeDim(name(X), (;values=nothing))
# keep-as-provided: a collection value is a real axis (iterate its coords); a scalar,
# `missing` (unlabelled) or `nothing` (aggregated) value is a single fixed position.
_coords(v::Union{Tuple,AbstractArray,AbstractRange}) = v
_coords(v) = (v,)
Base.length(X::TreeDim) = length(_coords(meta(X).values))
Base.iterate(X::TreeDim, args...) = iterate(_coords(meta(X).values), args...)
struct TreeData{P,M<:NamedTuple}
    parent::P
    meta::M
    TreeData(parent, meta::NamedTuple) = new{typeof(parent), typeof(meta)}(parent, meta)
end
Base.parent(X::TreeData) = getfield(X, :parent)
meta(X::TreeData) = getfield(X, :meta)
TreeData(dims::Symbol...) = (x, vals...)->TreeData(x, map(TreeDim, dims, vals)...)
TreeData(x, dims...; kwargs...) = TreeData(x, map(TreeDim, dims)..., map(TreeDim, keys(kwargs), values(kwargs))...)
TreeData(x, dims::TreeDim...) = TreeData(x, (;dims))
# a TreeNamedTuple keeps its record axis in the type as a SEPARATE `outer_dim` (which axis
# enumerates the fields is a property of the container, not of a dim) *and* lists it in
# `dims` like any other axis, so it's not lost when forwarded (e.g. by `TreeData(X::TreeData, dims...)`).
TreeData((name, X)::Pair{Symbol,<:NamedTuple}, dims::TreeDim...) = begin
    rec = TreeDim(name, keys(X))
    TreeData(X, (;dims = (dims..., rec), outer_dim = rec))
end
TreeData(X::TreeData, dims::TreeDim...) = TreeData(parent(X), merge(meta(X), (;dims=(meta(X).dims..., dims...))))
TreeNamedTuple{P<:NamedTuple,M<:NamedTuple} = TreeData{P,M}
TreeRaggedArray{P<:AbstractArray{<:TreeData},M<:NamedTuple} = TreeData{P,M}
TreeArray{P<:AbstractArray,M<:NamedTuple} = TreeData{P,M}
TreeTuple{P<:Tuple,M<:NamedTuple} = TreeData{P,M}
# convenience accessors (avoid spelling out `meta(...).field` everywhere)
# `dims` collides with the `dims=` kwarg used throughout mapslices/quantile,
# so inside those method bodies it must be qualified as `Main.dims(...)`
# (correct only while this script lives in `Main`) -- once this code moves
# into the TreeArrays module, requalify those call sites as `TreeArrays.dims`.
dims(X::TreeData) = meta(X).dims                 # the tree's (inner) axes
outerdim(X::TreeData) = meta(X).outer_dim        # a TreeNamedTuple's record axis

# leaf numeric eltype: recurse through NamedTuple / ragged nesting down to the backing array.
# TreeNamedTuple uses the FIRST field's type -- fine for quantile's homogeneous numeric records.
_eltype(X::TreeArray)       = eltype(parent(X))
_eltype(X::TreeTuple)       = eltype(parent(X))
_eltype(X::TreeNamedTuple)  = _eltype(first(parent(X)))
_eltype(X::TreeRaggedArray) = _eltype(first(parent(X)))
_eltype(x)                  = eltype(x)

# ===================== TreeArray (array-backed) array interface =====================
# TreeData is deliberately not an AbstractArray (TreeNamedTuple / TreeRaggedArray aren't
# arrays either); only the array-backed TreeArray alias gets the basic array interface,
# delegating straight through to the backing array. TreeRaggedArray{P<:AbstractArray{<:TreeData}}
# is a strict subconstraint of TreeArray{P<:AbstractArray}, so these methods also apply to
# ragged values at the OUTER level (size/length -> outer count, getindex/iterate -> sub-trees) --
# intentional, sensible outer-level semantics, not a gap.
Base.size(X::TreeArray, args...) = size(parent(X), args...)
Base.length(X::TreeArray) = length(parent(X))
Base.ndims(X::TreeArray) = ndims(parent(X))
Base.eltype(T::Type{<:TreeArray}) = eltype(fieldtype(T, :parent))
Base.axes(X::TreeArray, args...) = axes(parent(X), args...)
Base.getindex(X::TreeArray, i...) = getindex(parent(X), i...)
Base.iterate(X::TreeArray, args...) = iterate(parent(X), args...)
Base.collect(X::TreeArray) = collect(parent(X))


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
print_values(io::IO, X::TreeData) = if get(io, :compact, false)
    print(io, parent(X))
else
    print(io, "\n")
    print(IOContext(io, :compact => true), parent(X))
end
print_values(io::IO, X::TreeNamedTuple) = if get(io, :compact, false)
    print(io, parent(X))
else
    print(io, "\n")
    io = IOContext(io, :compact => true)
    for (k, v) in pairs(parent(X))
        print(io, k, ": ", v, "\n")
    end
end
Base.show(io::IO, D::TreeDim) = begin
    print(io, name(D), ": ")
    print(IOContext(io, :compact => true), meta(D).values)
end

# ===================== dim helpers =====================
# an *axis* dim backs real structure (array axis / NT keys / ragged nesting): its value
# is `missing` (unlabelled) or a collection. A scalar value = a fixed-position singleton;
# `nothing` = an aggregated ("sliced") ghost. Neither backs an axis.
_isaxis(d::TreeDim) = _isaxis(meta(d).values)
_isaxis(::Union{Tuple,AbstractArray,AbstractRange}) = true   # array / NT / ragged axis
_isaxis(::Missing) = true                                    # unlabelled, but still an axis
_isaxis(_) = false                                           # scalar (fixed) or nothing (sliced)
_dimnames(d::Symbol) = (d,)
_dimnames(dims) = Tuple(dims)
_aschild(v::TreeData, inner) = v                 # already a TreeData -> knows its own dims
_aschild(v, inner) = TreeData(v, inner...)       # raw field -> wrap with the inner axes

# ===================== mapslices =====================
# Reduce the named `dims`: apply `f` to each leftover-index slice, keep everything.
# `f` returns a TreeData (or a scalar). Reduced dims stay but become `sliced` (aggregated).
# A requested dim that is absent from a leaf -> `missing` (fixed sentinel).

# Shared "found the axis" bookkeeping for TreeArray/TreeRaggedArray: which parent-array
# positions are being reduced (redaxes) vs kept (keepaxes), and the ghost dims left behind
# for the reduced ones (trailing). Returns `nothing` when `want` doesn't hit a real axis
# here -> the caller decides what that means (a true leaf -> sentinel/idempotent re-slice;
# an intermediate node -> recurse deeper into each element).
function _reduceouter(f, X, want)
    alldims = Main.dims(X)
    names   = map(name, alldims)
    n_ax    = ndims(parent(X))
    @assert all(_isaxis, alldims[1:n_ax]) "_reduceouter assumes the first $n_ax dims of $(typeof(X)) are exactly the parent array's axes, positionally, in order; got a non-axis dim at position $(findfirst(!_isaxis, alldims[1:n_ax])) (dims = $(names))"
    redaxes = Tuple(i for i in 1:n_ax if names[i] in want)
    isempty(redaxes) && return nothing
    keepaxes = Tuple(i for i in 1:n_ax if !(names[i] in want))
    keptdims = Tuple(alldims[i] for i in keepaxes)
    keptset  = Set(keepaxes)
    trailing = Tuple(name(d) in want ? sliced(d) : d for (i, d) in enumerate(alldims) if !(i in keptset))
    outs = isempty(keepaxes) ? _leafreduce(f, parent(X)) : map(sl -> _leafreduce(f, sl), eachslice(parent(X); dims=keepaxes))
    _assemble(outs, keptdims, trailing)
end

# The gather-reduction: reduce an OUTER axis of a nested result by pushing it down to the
# leaves (pure index arithmetic on `sl`, a gathered slice along that axis -- nothing
# materialized) and recursing. Record (NamedTuple- or Tuple-keyed) -> recurse per key.
# Array-backed leaf -> recurse per position, building a NEW nested array (never
# flattened/stacked -- a chained reduction stays a tree of arrays all the way down).
# Scalar-backed leaf -> no inner positions to preserve; the outer reduction replaces the
# leaf outright (mirrors the plain-array base case below). Plain array leaf -> apply the
# kernel directly; this base case is also what a dense TreeArray reduction needs, so
# `_reduceouter` routes all shapes through `_leafreduce` uniformly.

# TreeNamedTuple (named fields) and TreeTuple (positional fields) are both "keyed, axis-
# bearing" records and walk the identical gather-by-key / recurse / reassemble shape; only
# how keys are read and how the container is rebuilt differs -- shared driver below, two
# tiny per-container dispatch points instead of a bespoke tuple-only branch.
_reassemble(::NamedTuple{ks}, fields) where ks = NamedTuple{ks}(fields)
_reassemble(::Tuple, fields)                   = Tuple(fields)
_leafmeta(proto::TreeNamedTuple) = (;dims = Main.dims(proto), outer_dim = outerdim(proto))
# a TreeTuple leaf never carries an `outer_dim` (positional fields have no naming axis to
# record); if a future path builds a TreeTuple WITH one, this drops it silently -- widen
# this method (not a bespoke special case) if that ever becomes real.
_leafmeta(proto::TreeTuple)      = (;dims = Main.dims(proto))
_leafreduce(f, sl::AbstractArray{<:Union{TreeNamedTuple,TreeTuple}}) = begin
    proto = first(sl)
    ks = keys(parent(proto))
    fields = map(k -> _leafreduce(f, map(el -> parent(el)[k], sl)), ks)
    TreeData(_reassemble(parent(proto), fields), _leafmeta(proto))
end
_leafreduce(f, sl::AbstractArray{<:TreeData}) = begin
    proto = first(sl)
    p = parent(proto)
    p isa AbstractArray || return _leafreduce(f, map(parent, sl))  # scalar leaf: reduce directly, no positions
    vals = map(i -> _leafreduce(f, map(el -> parent(el)[i], sl)), CartesianIndices(p))
    TreeData(vals, meta(proto))
end
_leafreduce(f, sl::AbstractArray) = f(sl)

function Base.mapslices(f, X::TreeArray; dims)
    want = _dimnames(dims)
    r = _reduceouter(f, X, want)
    isnothing(r) || return r
    alldims = Main.dims(X)
    any(nm -> nm in map(name, alldims), want) || return missing   # dim absent here -> sentinel
    TreeData(parent(X), (;dims = map(d -> name(d) in want ? sliced(d) : d, alldims)))
end

function Base.mapslices(f, X::TreeNamedTuple; dims)
    want = _dimnames(dims)
    rec  = outerdim(X)
    inner  = Tuple(d for d in Main.dims(X) if _isaxis(d) && name(d) != name(rec))   # rec enumerates the fields, not an inner axis
    ghosts = Tuple(d for d in Main.dims(X) if !_isaxis(d))
    newfields = map(v -> mapslices(f, _aschild(v, inner); dims), parent(X))
    any(!ismissing, newfields) || return missing        # no child carried the dim -> sentinel
    sample = first(v for v in newfields if !ismissing(v))
    have   = map(name, Main.dims(sample))
    extra  = Tuple(g for g in ghosts if !(name(g) in have))
    TreeData(newfields, (;dims = (Main.dims(sample)..., extra...), outer_dim = rec))
end

function Base.mapslices(f, X::TreeRaggedArray; dims)
    want = _dimnames(dims)
    r = _reduceouter(f, X, want)
    isnothing(r) || return r
    TreeData(map(el -> mapslices(f, el; dims), parent(X)), meta(X))
end

# wrap `outs` (the raw per-slice kernel outputs -- already TreeData/record/scalar pieces,
# never stacked/pivoted) as one TreeData over the kept + reduced-as-ghost dims.
_assemble(outs, keptdims, trailing) = TreeData(outs, (;dims = (keptdims..., trailing...)))

# ===================== reducers =====================
Statistics.mean(X::TreeData; dims=nothing) = isnothing(dims) ? mean(parent(X)) : mapslices(mean, X; dims)
Base.sum(X::TreeData; dims=nothing) = isnothing(dims) ? sum(parent(X)) : mapslices(sum, X; dims)

# quantile delegates to mapslices; pdim bundles the output axis name + values,
# kept EXACTLY as given (Tuple stays Tuple, Vector stays Vector, Number stays
# Number) -- fed RAW to quantile!, which mirrors the same container back into
# the leaf (decision xxmv6c option 2). The tree machinery (_leafreduce) handles
# the resulting Tuple- and scalar-parent leaves directly, including when a
# quantile leaf feeds a CHAINED quantile call. One shared sort buffer.
function Statistics.quantile(X::TreeData, pdim::TreeDim; dims)
    p      = meta(pdim).values
    scratch = _eltype(X)[]
    mapslices(X; dims) do slice
        length(scratch) == length(slice) || resize!(scratch, length(slice))
        copyto!(scratch, slice)
        TreeData(quantile!(scratch, p), pdim)
    end
end

# ===================== dimension-aware kernels =====================
# Annotate a per-slice kernel with its dim-signature `reduces => into`. Two forms:
#   defining:  @kernel (:time => :stat) function f(L) ... end   -- define the plain array
#              kernel AND a TreeData method that threads it through mapslices.
#   post hoc:  @kernel (:time => :stat) f                        -- just add the TreeData
#              method to an already-defined f (e.g. a fixed-signature function you reuse).
# The output structure is inferred from the kernel's return type (the _assemble dispatch);
# the annotation only supplies the consumed dim + the record name. Parameterized reducers
# (mean/sum reduce->scalar, quantile reduce->axis, dim/into chosen per-call) DON'T fit this
# fixed-signature shape -- they stay as mapslices-delegating methods with dims=/into=.
macro kernel(spec, fdef)
    reduces = spec.args[2]                                   # consumed dim, e.g. :time
    into    = spec.args[3]                                   # produced record axis, e.g. :stat
    name    = fdef isa Symbol ? fdef : fdef.args[1].args[1]  # bare name (post hoc) or a def
    treemethod = :($name(X::TreeData) = mapslices(s -> TreeData($into => $name(s)), X; dims=$reduces))
    # a def -> also emit the plain array kernel; a bare name -> just wrap an existing function.
    esc(fdef isa Symbol ? treemethod : Expr(:block, fdef, treemethod))
end

unsetdim(X) = X
setdim(X::TreeData; kwargs...) = error("setdim not implemented")#TreeData(unsetdim(parent(X); kwargs...), (;dims=setdim(meta(X).dims; kwargs...)))
setdim(dims::Tuple; kwargs...) = error("setdim not implemented")#values(merge(), (;kwargs...))

Base.cat(X::TreeData...) = TreeData(X)
Base.stack(f, iter::TreeDim) = map(f, iter)#
Base.stack(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{<:TreeDim}}}) = map(f, iter)
Base.map(f, iter::TreeDim) = TreeData(map(f, meta(iter).values), iter)
Base.map(f, iter::Base.Iterators.ProductIterator{<:Tuple{<:TreeDim, Vararg{<:TreeDim}}}) = TreeData(
    map(f, Iterators.product((meta(iter).values for iter in iter.iterators)...)), iter.iterators...
)


begin
    # # Should actually be doing some stuff to the X matrix - not doing it here for now
    # zero_re(X::TreeData) = setdim(X; random_effect=:zero)
    # # Should actually be doing some stuff to the X matrix - not doing it here for now, but maybe this will be the first change
    # constrain(X::TreeData) = setdim(X, dims.space.user)
    # # Should actually be doing some stuff to the X matrix - not doing it here for now
    # noplacebo(X::TreeData) = setdim(X, dims.placebo.off)
    # # Should actually be doing some stuff to the X matrix - not doing it here for now
    # setschedule(X::TreeData, schedule) = setdim(X, dims.schedule=>schedule)
    dense_loc(args...) = TreeData(
        randn(n_draws, n_subjects, n_dense),
        :draw, :subject, :time=>range(0, 1, n_dense)
    )
    # compute_stats: a dimension-aware kernel. Written once as pure math on a time-slice,
    # annotated `:time => :stat` -> also gets a TreeData method that reduces :time into a
    # :stat record. `compute_stats(::AbstractArray)` and `compute_stats(::TreeData)` both work.
    @kernel (:time => :stat) function compute_stats(L)
        trough, peak = extrema(L)
        baseline = L[1]
        dtrough, dpeak = extrema(L .- baseline)
        (;trough, peak, baseline, dtrough, dpeak)
    end


    n_subjects = 179
    healthy = rand(Bool, n_subjects)
    diseased = .!healthy
    male = rand(Bool, n_subjects)
    female = .!male
    weight = randn(n_subjects)
    age = randn(n_subjects)
    n_doses = [rand(1:28) for _ in 1:n_subjects]
    dose_times = map(sort ∘ randn, n_doses)
    dose_amounts = map(randn, n_doses)
    n_measurements = [rand(2:100) for _ in 1:n_subjects]
    measurement_times = map(sort ∘ randn, n_measurements)
    measurement_values = map(randn, n_measurements)
    # input_data = (;
    #     healthy, male, weight, age,
    #     dose_times, dose_amounts,
    #     measurement_times, measurement_values
    # )
    input_data = TreeData(
        :data=>(;
            healthy, diseased, male, female, weight, age,
            dose=map(TreeData(:time), dose_amounts, dose_times),
            measurement=map(TreeData(:time), measurement_values, measurement_times)
        ),
        :subject=>1:n_subjects
    )
    display(input_data)

    # reduce over :time through the heterogeneous, ragged tree: fields lacking a
    # :time axis come back `missing`; the ragged dose/measurement series are reduced.
    display(mapslices(mean, input_data; dims=:time))

    n_draws = 1000
    n_dense = 100
    sweep_schedules = TreeDim(:schedule, ("some schedule", ))
    sweep_doses = TreeDim(:dose, (20, 200))
    population_quantiles = (0.05, 0.5, 0.95)   # placeholder levels: summarize across subjects
    posterior_quantiles = (0.05, 0.5, 0.95)    # placeholder levels: summarize across draws

    n_params = (;
        baseline=6+n_subjects,
        dose=6+n_subjects,
        placebo=3,
        effect=4,
        noise=1,
        other=2
    )
    n_cols = sum(n_params)

    input_draws = TreeData(
        randn(n_draws, n_cols),
        :draw, :param; random_effect=:in_sample, placebo=:on, space=:sampler
    )
    stats_percentiles = map(Iterators.product(
        TreeDim(:random_effect, (:zero, :population)),
        TreeDim(:placebo, (:on, :off)),
        sweep_schedules,
        sweep_doses
    )) do args...
        quantile(
            quantile(
                compute_stats(dense_loc(input_draws, args...)),
                TreeDim(:population, population_quantiles); dims=:subject
            ),
            TreeDim(:posterior, posterior_quantiles); dims=:draw
        )
    end

    display(stats_percentiles)

    # demo: scalar-p quantile mirrors Base — the pdim's fixed SCALAR value
    # becomes a fixed coordinate (not a length-1 axis), and each slice holds
    # one scalar value (not a length-1 vector).
    median_draws = quantile(input_draws, TreeDim(:median, 0.5); dims=:draw)
    display(median_draws)
end

# ===================== PROBE: sum(X) default + TreeArray array interface =====================
begin
    X = TreeData(randn(4,3), :draw, :param)
    P = parent(X)
    @assert sum(X) == sum(P)                # was: UndefKeywordError: dims
    @assert size(X) == size(P)              # was: MethodError
    @assert length(X) == length(P)          # was: MethodError
    @assert ndims(X) == ndims(P)            # was: MethodError
    @assert eltype(X) == eltype(P)          # was: Any
    @assert X[1] == P[1]                    # was: MethodError (no getindex)
    @assert collect(X) == collect(P)        # was: MethodError (no length/iterate)
    println("PROBE sum(X) = ", sum(X))
    println("PROBE size(X) = ", size(X), ", length(X) = ", length(X), ", ndims(X) = ", ndims(X), ", eltype(X) = ", eltype(X))
    println("PROBE X[1] = ", X[1])
    println("PROBE collect(X) == parent(X): ", collect(X) == P)
end

# ===================== PROBE: _reduceouter positional-axis assertion (jz9bkv) =====================
# _reduceouter assumes the first ndims(parent(X)) entries of `dims` are exactly the
# parent array's axes, positionally, in order (_assemble upholds this by construction,
# but nothing enforces it on a hand-built TreeData). A non-axis dim (here a scalar,
# `:extra`) placed BEFORE the real axes shifts the positional lookup and silently
# reduces the wrong parent axis -- confirmed below before landing the @assert:
#   reducing :draw on a well-formed (draw, param) TreeData sums over rows -> [10, 26, 42]
#   reducing :draw on the misordered (extra, draw, param) TreeData instead summed over
#   columns -> [15, 18, 21, 24] (wrong axis, wrong shape, silently wrong).
# Now that _reduceouter asserts the invariant, the misordered construction throws.
begin
    Pgood = reshape(1.0:12.0, 4, 3)
    Xgood = TreeData(Pgood, :draw, :param)
    Xbad  = TreeData(Pgood, TreeDim(:extra, 99), TreeDim(:draw), TreeDim(:param))
    good  = mapslices(sum, Xgood; dims=:draw)
    @assert parent(good) == [10.0, 26.0, 42.0]   # sane baseline: reduces rows, not columns
    threw = try
        mapslices(sum, Xbad; dims=:draw)
        false
    catch e
        e isa AssertionError || rethrow()
        true
    end
    @assert threw "expected AssertionError: non-axis dim before a real array axis must fail loudly, not mis-map"
    println("PROBE _reduceouter positional-axis assertion fired for misordered dims")
end

# ===================== PROBE: _leafreduce preserves multidim leaf shape (1cn9zad) =====================
# _leafreduce(f, sl::AbstractArray{<:TreeData}) built `vals` via
# map(i -> ..., eachindex(parent(proto))) -- eachindex is linear, so a multidim leaf's
# `vals` came back as a flat Vector while the reused `meta(proto)` still described the
# original multidim shape (confirmed below before the fix: meta claimed (:time, :chan)
# while parent(inner) was a flat length-6 Vector{Float64}). Fixed by iterating
# CartesianIndices(parent(proto)) instead of eachindex, which preserves the leaf's shape.
begin
    leaves = [TreeData(randn(2, 3), :time, :chan) for _ in 1:5]
    Xouter = TreeData(leaves, :subject)          # TreeRaggedArray: outer :subject axis
    result = mapslices(mean, Xouter; dims=:subject)
    inner  = parent(result)                      # the gathered per-leaf-cell TreeData
    @assert size(parent(inner)) == (2, 3) "leaf shape flattened: got $(size(parent(inner)))"
    println("PROBE _leafreduce preserves multidim leaf shape: ", size(parent(inner)))
end

# ===================== PROBE: setdim stubs throw instead of silently no-oping =====================
begin
    X = TreeData(randn(4,3), :draw, :param)
    @assert (try; setdim(X; foo=:bar); false; catch; true; end)                # was: silently returned X unchanged
    @assert (try; setdim((:draw, :param); foo=:bar); false; catch; true; end)  # was: bare error() with no message
end
# begin

# ===================== PROBE: outer_dim survives TreeData-forwarding =====================
begin
    Xnt = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(4), :draw)))
    Y = TreeData(Xnt, TreeDim(:extra, nothing))
    @assert outerdim(Y) === outerdim(Xnt)                    # was: no `outer_dim` field in meta(Y)
    Z = mapslices(mean, Y; dims=:draw)                       # was: errors calling outerdim(Y) inside mapslices (~line 205)
    println("PROBE outerdim(Y) = ", outerdim(Y))
    println("PROBE mapslices(mean, Y; dims=:draw) ran without error")
end
# begin

# ===================== PROBE: quantile(X, pdim::TreeDim) leaf shape mirrors pdim's values =====================
# The unified quantile no longer coerces p at all -- Base's quantile!(scratch, p)
# preserves p's container as-is (scalar -> scalar leaf, Vector -> Vector leaf, Tuple ->
# Tuple leaf), matching decision xxmv6c (option 2): a leaf's data mirrors p's container.
# The tree machinery (_leafreduce) was extended to gather-reduce non-array-backed leaves
# (Tuple- and scalar-parent) directly, via a path SHARED with TreeNamedTuple (both are
# keyed, axis-bearing records) rather than array-ifying the leaf. Before this fix, a
# Tuple pdim was `collect`-ed to force an array-backed leaf so chained quantile wouldn't
# hit _leafreduce's CartesianIndices; now the Tuple leaf survives the chain unchanged:
#   MethodError: no method matching CartesianIndices(::Tuple{Float64, Float64, Float64})
# -- confirmed live before landing this probe. A scalar-parent leaf hits the identical
# CartesianIndices(::Number) MethodError when chained -- also fixed, also probed here.
begin
    Xq = TreeData(reshape(1.0:12.0, 4, 3), :draw, :param)

    scalar_result = quantile(Xq, TreeDim(:median, 0.5); dims=:draw)
    scalar_leaf = parent(scalar_result)[1]
    @assert parent(scalar_leaf) isa Number "expected a scalar leaf for scalar p, got $(typeof(parent(scalar_leaf)))"
    @assert !_isaxis(dims(scalar_leaf)[1])       # fixed coordinate, not an axis

    vector_result = quantile(Xq, TreeDim(:pct, [0.25, 0.5, 0.75]); dims=:draw)
    vector_leaf = parent(vector_result)[1]
    @assert parent(vector_leaf) isa AbstractVector "expected a Vector leaf for a Vector p, got $(typeof(parent(vector_leaf)))"
    @assert _isaxis(dims(vector_leaf)[1])        # real axis

    tuple_result = quantile(Xq, TreeDim(:tup, (0.25, 0.5, 0.75)); dims=:draw)
    tuple_leaf = parent(tuple_result)[1]
    @assert meta(dims(tuple_leaf)[1]).values isa Tuple   # axis label: Tuple stays Tuple
    @assert parent(tuple_leaf) isa Tuple "expected a Tuple leaf for a Tuple p, got $(typeof(parent(tuple_leaf)))"

    chained = quantile(tuple_result, TreeDim(:tup2, (0.1, 0.9)); dims=:param)  # was: MethodError CartesianIndices(::Tuple)
    @assert parent(parent(chained)) isa Tuple "expected the chained result to stay a Tuple leaf, got $(typeof(parent(parent(chained))))"

    chained_scalar = quantile(scalar_result, TreeDim(:median2, 0.5); dims=:param)  # was: MethodError CartesianIndices(::Number)
    @assert parent(parent(chained_scalar)) isa Number "expected the chained scalar result to stay a scalar leaf"

    println("PROBE quantile(pdim) leaf shape mirrors p: scalar -> scalar leaf, Vector -> Vector leaf, Tuple -> Tuple leaf -- all survive chaining")
end
# begin



#     n_timepoints = 100
#     n_doses = 12

#     dims = map()

#     dims = (;
#         draw=(;
#             groups=(;
#                 chains=1,
#             )
#         ),
#         param=(;
#             groups=(;
#                 baseline=(;
#                     fixed=(;
#                         intercept=1,
#                         zage=2,
#                         zweight=3,
#                         male=4,
#                         diseased=5,
#                     ),
#                     log_scale=6,
#                     random=7:6+n_subjects
#                 )
#             )
#         ),
#         time=(;
#             # value=range(0, 1, n_draws),
#         ),
#         schedule=(;

#         ),
#         dose=(;
#             value=exp.(range(0, 1, n_doses))
#         ),
#         subject=(;
#             value=1:n_subjects,
#             groups=(;healty, diseased, male, female)
#         ),
#         random_effect=(;
#             value=(:in_sample, :zero, :population)
#         ),
#         placebo=(;
#             value=(:on, :off)
#         ),
#         space=(;value=(:sampler, :user))
#     )

#     S = TreeData(
#         randn(n_draws, n_cols),
#         (;dims=(
#             dims.draw,
#             dims.param,
#             dims.random_effect.in_sample,
#             dims.placebo.on,
#             dims.space.sampler
#         ))
#     )
#     S0 = zero_re(S)
#     SS0 = stack((S, S0))
#     UU0 = constrain(SS0)
#     R2 = mapslices(UU0; dims=(:subject, :random_effect)) do sub
#         tmp = var(sub; dims=:subject)
#         tmp[tmp.random_effect.zero] ./ tmp[tmp.random_effect.in_sample]
#     end



#     P = setschedule(setre(S, :population), "something")
#     Ls = stack(Iterators.product(doses, placebos)) do dose, placebo
#         loc(setplacebo(setdose(P, dose), placebo))
#     end

#     stats = compute_stats(Ls)
#     stats_percentiles = quantile(quantile(stats, population_quantiles; dims=subject), posterior_quantiles; dims=draw)

#     dLs = mapslices(Ls; dims=time) do L
#         setdim(L .- L[1]; qoi="d" * L.qoi)
#     end
#     LdLs = stack((Ls, dLs))
#     LdLs_percentiles = quantile(quantile(LdLs, population_quantiles; dims=subject), posterior_quantiles; dims=draw)



# end
