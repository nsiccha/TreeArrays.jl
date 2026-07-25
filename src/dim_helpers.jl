# ===================== dim helpers =====================
# an *axis* dim backs real structure (array axis / NT keys / ragged nesting): its value
# is `missing` (unlabelled) or a collection. A scalar value = a fixed-position singleton;
# `nothing` = an aggregated ("sliced") ghost. Neither backs an axis.
_isaxis(d::TreeDim) = _isaxis(meta(d).values)
_isaxis(::Union{Tuple,AbstractArray,AbstractRange}) = true   # array / NT / ragged axis
_isaxis(::Missing) = true                                    # unlabelled, but still an axis
_isaxis(_) = false                                           # scalar (fixed) or nothing (sliced)
# type-level mirror: the instance-level dispatch above already only depends on the *type*
# of `meta(d).values`, never its content, so this is a direct extension (not a parallel
# implementation) -- lets `@generated` helpers classify a dims tuple's element *types*
# (e.g. `alldims.parameters[i]`) without materializing instances.
_isaxis(::Type{<:TreeDim{N,M}}) where {N,M} = _isaxis(fieldtype(M, :values))
_isaxis(::Type{<:Union{Tuple,AbstractArray,AbstractRange}}) = true
_isaxis(::Type{Missing}) = true
_isaxis(::Type) = false
_dimnames(d::Symbol) = (d,)
_dimnames(dims) = Tuple(dims)

# ===================== `coords` -- read an axis's coordinates =====================
# The public accessor for what an axis is LABELLED with. Without it a consumer holding a tree
# had no way to ask: a `TreeDim`'s only field is `meta` (so `d.values` is a plain `getfield`
# failure -- nothing intercepts `getproperty` here), and `meta`/`name` are internal.
#
# `collect(d)` LOOKS like the answer -- `TreeDim` has `iterate`/`length`, so it runs -- and is a
# TRAP twice over: it yields an `Any`-eltype vector (discarding exactly the inference the
# function barrier is for), and it silently answers `Any[missing]` for an UNLABELLED axis and
# `Any[nothing]` for a reduced GHOST -- a plausible-looking length-1 coordinate vector for the
# two cases that have NO coordinates at all. That is what 16fwcnx forbids, so the cases with no
# answer error BY NAME here, each naming the spelling that would have one.
#
# NB `_coords` (types.jl) is the INTERNAL iteration helper and deliberately does the OPPOSITE:
# it wraps a scalar / `missing` / `nothing` into a 1-tuple so `iterate`/`length` stay total over
# every dim kind. Public `coords` answers the consumer's question instead, and refuses what has
# no answer. This is also what `coords=true` binds through (mapslices.jl), so a kernel and a
# consumer now read coordinates through ONE definition with ONE error message.
coords(d::TreeDim) = _coordvals(name(d), meta(d).values)
_coordvals(::Symbol, values::Union{Tuple,AbstractArray,AbstractRange}) = values
_coordvals(n::Symbol, ::Missing) = error(
    "TreeArrays: dim `:$n` is unlabelled (values === missing), so it has no coordinates to read. " *
    "Give them at construction (`TreeData(x, :$n => labels)`).")
_coordvals(n::Symbol, ::Nothing) = error(
    "TreeArrays: dim `:$n` is an aggregated ghost -- it was REDUCED away, so it has no coordinates " *
    "left by construction. Read them off the tree BEFORE reducing `:$n`.")
_coordvals(n::Symbol, v) = error(
    "TreeArrays: dim `:$n` is a fixed single position (value `$v`), not a labelled axis, so it has " *
    "no coordinate vector -- returning `$v` would lie about its shape. That one value IS its label; " *
    "pass a 1-tuple (`:$n => ($v,)`) to make it a real length-1 axis.")

# by NAME on this node -- the ergonomic form, foundALL like `dims=` (a name that resolves
# nowhere is a typo, not an empty answer).
#
# `@generated` position lookup, for the same reason as `_splitdims`/`_reducedaxes`: a runtime
# `findfirst` over a HETEROGENEOUS dims tuple makes the return type the UNION of every dim's
# coordinate type (measured: `Union{Vector{Float64},Vector{Symbol}}` on a `(:time,:subject)`
# tree), so a plain loop would hand a consumer an inference-opaque read. Keyed on the dims TYPE
# + `Val{nm}` it folds to one `getindex`. `@constprop :aggressive` is what lets the ordinary
# `coords(X, :time)` call site -- a literal Symbol, not a `Val` -- reach the staged lookup.
Base.@constprop :aggressive coords(X::TreeData, nm::Symbol) =
    coords(_dimat(TreeArrays.dims(X), Val(nm)))

# the loop is spelled out rather than reusing `_dimtypes` (mapslices.jl): that file is included
# LATER, and a self-contained generator keeps this free of any include-order question.
@generated function _dimat(alldims::Tuple, ::Val{nm}) where nm
    ps = alldims.parameters
    for i in 1:length(ps)
        name(ps[i]) === nm && return :(alldims[$i])
    end
    msg = "TreeArrays: no dim named `:$nm` on this node -- it carries $(map(name, Tuple(ps))). " *
          "A dim living inside the LEAVES of a ragged tree is on the child, not here: reach it " *
          "via that child (`coords(parent(X)[i], :$nm)`)."
    :(error($msg))
end
_aschild(v::TreeData, inner) = v                 # already a TreeData -> knows its own dims
_aschild(v, inner) = TreeData(v, inner...)       # raw field -> wrap with the inner axes
