# ===================== broadcasting keeps the labels (decision 0jwqiy5) =====================
# `exp.(X)`, `X .- 1`, `post.lr .* 2`, `X .- mean(X; dims = :draw)` return a TREE with the
# same dims. Before, a TreeArray broadcast through its forwarded array interface and came back
# as a plain `Array`, dropping every label without a word.
#
# The rules (the user's pick on 0jwqiy5, prong `broadcast`):
#   - Scalars, `Ref`s and other broadcast-scalars combine with any tree.
#   - A record broadcasts field by field; a ragged or positional tree leaf by leaf (both sides
#     must share the outer axis: same name, length and labels).
#   - At a leaf, the operand with the most axes is the reference. Every other tree must carry
#     those axes BY NAME, in the same order, with the same lengths and labels. An axis it lacks
#     must be one of its REDUCED or FIXED dims -- that is how `X .- mean(X; dims = :draw)`
#     centres the draws: the mean keeps `:draw` as a reduced dim and stretches along it.
#   - Anything else refuses by name: an axis present in one tree and absent from another, a
#     different order, different labels, and plain arrays (whose axes have no names to match).
# The result carries the reference's dims. Computation is Base's broadcast over the backing
# arrays (a reduced operand is a `reshape` view, never a copy).

struct _TreeStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:TreeData}) = _TreeStyle()
Base.BroadcastStyle(::_TreeStyle, ::_TreeStyle) = _TreeStyle()
Base.BroadcastStyle(s::_TreeStyle, ::Broadcast.BroadcastStyle) = s
Base.broadcastable(X::TreeData) = X
# A tree has no single `axes` to combine (a record or ragged node has none at all); the rules
# above do the shape check, so skip Base's axes pass.
Broadcast.instantiate(bc::Broadcast.Broadcasted{_TreeStyle}) = bc
function Base.copy(bc::Broadcast.Broadcasted{_TreeStyle})
    fl = Broadcast.flatten(bc)            # fuse nested dot-calls into one function of flat args
    _tbcnode(fl.f, map(_bcarg, fl.args))
end

_bcarg(x) = x
_bcarg(x::AbstractArray) = ndims(x) == 0 ? x : _refusebcarray(x)
_bcarg(x::Tuple) = length(x) == 1 ? x : _refusebcarray(x)
_refusebcarray(x) = throw(ArgumentError(
    "TreeArrays: cannot broadcast a tree together with a plain " * string(typeof(x).name.name) *
    " of size " * string(x isa Tuple ? (length(x),) : size(x)) * ": its axes have no names to match the " *
    "tree's by. Wrap it as a tree with named dims (`TreeData(x, :participant => ids)`), or pass it " *
    "whole to every element with `Ref(x)`."))

_iscontainer(a) = a isa TreeRaggedArray || a isa TreeTuple

function _tbcnode(f, args::Tuple)
    i = findfirst(a -> a isa TreeNamedTuple, args)
    i === nothing || return _tbcrecord(f, args, args[i])
    j = findfirst(_iscontainer, args)
    j === nothing || return _tbccontainer(f, args, args[j])
    _tbcleaf(f, args)
end

_recinner(a::TreeNamedTuple) = first(_splitrecord(TreeArrays.dims(a), Val(name(outerdim(a)))))
_bcfield(a::TreeNamedTuple, k) = _aschild(parent(a)[k], _recinner(a))
# A field result over exactly the record's inner dims goes back in raw, like the input stored it.
_unchild(v::TreeArray, inner) = isequal(TreeArrays.dims(v), inner) ? parent(v) : v
_unchild(v, inner) = v

function _tbcrecord(f, args, ref)
    ks = keys(parent(ref))
    inner = _recinner(ref)
    innames = map(name, inner)
    for a in args
        a isa TreeData || continue
        if a isa TreeNamedTuple
            keys(parent(a)) == ks || throw(ArgumentError(
                "TreeArrays: cannot broadcast records with different fields: " * string(ks) * " and " *
                string(keys(parent(a))) * "."))
        else
            extra = filter(n -> !(n in innames), map(name, _liveaxes(a)))
            isempty(extra) || throw(ArgumentError(
                "TreeArrays: cannot broadcast a record over " * string(innames) * " with a tree that has the " *
                "extra axes " * string(extra) * ": every field would gain axes the record does not carry. " *
                "Broadcast field by field (`X.field`), or reduce those axes first."))
        end
    end
    fields = map(k -> _unchild(_tbcnode(f, map(a -> a isa TreeNamedTuple ? _bcfield(a, k) : a, args)), inner), ks)
    TreeData(NamedTuple{ks}(fields), _leafmeta(ref))
end

function _tbccontainer(f, args, ref)
    P  = parent(ref)
    nd = ref isa TreeTuple ? 1 : ndims(P)
    outer = TreeArrays.dims(ref)[1:nd]
    for a in args
        a isa TreeData || continue
        _iscontainer(a) && typeof(parent(a)) <: (ref isa TreeTuple ? Tuple : AbstractArray) || throw(ArgumentError(
            "TreeArrays: cannot broadcast a " * (ref isa TreeTuple ? "positional" : "ragged") * " tree over " *
            string(map(name, outer)) * " together with a " * string(nameof(typeof(parent(a)))) * "-backed " *
            "tree: broadcasting goes leaf by leaf, so both must be split along the same outer axis."))
        Q = parent(a)
        ok = (ref isa TreeTuple ? length(Q) == length(P) : size(Q) == size(P)) &&
             all(((d, e),) -> name(d) === name(e) && _poolcoordsagree(meta(d).values, meta(e).values),
                 zip(TreeArrays.dims(a)[1:nd], outer))
        ok || throw(ArgumentError(
            "TreeArrays: cannot broadcast trees split along different outer axes: " *
            string(map(name, TreeArrays.dims(a)[1:nd])) * " vs " * string(map(name, outer)) *
            " (names, lengths and labels must all agree)."))
    end
    leaf(i) = _tbcnode(f, map(a -> a isa TreeData ? parent(a)[i] : a, args))
    out = ref isa TreeTuple ? ntuple(leaf, length(P)) : map(leaf, CartesianIndices(P))
    TreeData(out, meta(ref))
end

_liveaxes(a::TreeData) = parent(a) isa AbstractArray ? TreeArrays.dims(a)[1:ndims(parent(a))] : ()

function _tbcleaf(f, args)
    trees = filter(a -> a isa TreeData, args)
    ref = trees[argmax(map(a -> length(_liveaxes(a)), collect(trees)))]
    refax = _liveaxes(ref)
    out = broadcast(f, map(a -> a isa TreeData ? _bcprep(a, ref, refax) : a, args)...)
    TreeData(out, meta(ref))
end

function _bcprep(a::TreeData, ref, refax)
    p = parent(a)
    p isa Number && return p
    p isa AbstractArray || throw(ArgumentError(
        "TreeArrays: cannot broadcast a " * string(nameof(typeof(p))) * "-backed tree."))
    a === ref && return p
    ax = _liveaxes(a)
    refnames = map(name, refax)
    pos = map(d -> findfirst(==(name(d)), refnames), ax)
    for (d, j) in zip(ax, pos)
        j === nothing && throw(ArgumentError(
            "TreeArrays: cannot broadcast: `:" * string(name(d)) * "` is an axis of one tree but not of the " *
            "other (axes " * string(map(name, ax)) * " vs " * string(refnames) * ")."))
    end
    issorted(collect(pos)) || throw(ArgumentError(
        "TreeArrays: cannot broadcast trees whose shared axes come in a different order: " *
        string(map(name, ax)) * " vs " * string(refnames) * "."))
    for (k, (d, j)) in enumerate(zip(ax, pos))
        size(p, k) == size(parent(ref), j) && _poolcoordsagree(meta(d).values, meta(refax[j]).values) ||
            throw(ArgumentError(
                "TreeArrays: cannot broadcast: the `:" * string(name(d)) * "` axis differs between the trees " *
                "(lengths " * string(size(p, k)) * " vs " * string(size(parent(ref), j)) * ", or different labels)."))
    end
    adims = TreeArrays.dims(a)
    for (j, rd) in enumerate(refax)
        j in pos && continue
        i = findfirst(d -> name(d) === name(rd), adims)
        i !== nothing && !_isaxis(adims[i]) || throw(ArgumentError(
            "TreeArrays: cannot broadcast: `:" * string(name(rd)) * "` is an axis of one tree and absent from " *
            "the other. A tree stretches along an axis only where it carries it as a reduced or fixed dim " *
            "(e.g. `X .- mean(X; dims = :" * string(name(rd)) * ")`); otherwise rebuild it with that axis."))
    end
    reshape(p, ntuple(j -> (k = findfirst(==(j), pos); k === nothing ? 1 : size(p, k)), length(refax)))
end
