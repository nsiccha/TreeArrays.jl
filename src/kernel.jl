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
#
# COORDINATES. A kernel declaring TWO plain positional arguments additionally receives the
# reduced axis's COORDINATE vector (`coords=true`, mapslices.jl) -- the AUC / tmax shape:
#     @kernel (:time => :stat) function nca(y, t)
#         (; cmax = maximum(y), tmax = t[argmax(y)], auc = trapz(t, y))
#     end
# Inferring the opt-in from arity is safe HERE and only here: the macro reads the literal
# argument list at expansion time, so it never has to guess from a value the way a
# `hasmethod(f, (slice, coords))` probe would at the `mapslices` boundary -- where
# `maximum(f, itr)` makes that probe actively dangerous (see mapslices.jl).
#
# Only PLAIN positional arguments count. A default (`f(L, scale=2)`), keywords, and a splat
# (`f(L, rest...)`) are all 1-slice kernels with extra machinery, NOT coordinate consumers, so
# they keep today's behaviour instead of having a coordinate vector silently pushed into their
# second slot. The post-hoc form has no argument list to read at all, so it defaults to 1-arg
# and takes an EXPLICIT opt-in:  @kernel (:time => :stat) coords=true nca

# a definition's signature call-expr, past any `where` clause (`function f(x) where T`).
_unwhere(s) = s isa Expr && s.head === :where ? _unwhere(s.args[1]) : s
_kernelsig(fdef) = _unwhere(fdef.args[1])
# PLAIN positional argument count: `Expr(:kw, …)` (a default), `Expr(:parameters, …)` (keywords)
# and `Expr(:…, …)` (a splat) are deliberately not counted -- see the note above.
_kernelarity(sig) = count(a -> a isa Symbol || (a isa Expr && a.head === :(::)), sig.args[2:end])
_kernelcoordsopt(ex) =
    (ex isa Expr && ex.head === :(=) && ex.args[1] === :coords && ex.args[2] isa Bool) ? ex.args[2] :
    error("@kernel: expected `coords=true` or `coords=false` between the dim-signature and the " *
          "kernel, got `$ex`")

macro kernel(spec, args...)
    1 <= length(args) <= 2 || error(
        "@kernel: expected `@kernel (:dim => :into) [coords=true] <function definition or name>`")
    reduces  = spec.args[2]                                   # consumed dim, e.g. :time
    into     = spec.args[3]                                   # produced record axis, e.g. :stat
    fdef     = last(args)
    explicit = length(args) == 2 ? _kernelcoordsopt(args[1]) : nothing
    if fdef isa Symbol                                        # post hoc: no argument list to read
        nm, arity = fdef, nothing
    else
        sig = _kernelsig(fdef)
        nm, arity = sig.args[1], _kernelarity(sig)
        arity in (1, 2) || error(
            "@kernel: a kernel takes the data slice, optionally followed by the reduced axis's " *
            "coordinates -- `$nm` declares $arity plain positional arguments")
        explicit === true && arity == 1 && error(
            "@kernel: `coords=true` on `$nm`, which takes ONE positional argument -- a " *
            "coordinate-aware kernel is `function $nm(slice, coords)`")
        explicit === false && arity == 2 && error(
            "@kernel: `coords=false` on `$nm`, which takes TWO positional arguments -- drop the " *
            "second argument, or say `coords=true`")
    end
    treemethod = (isnothing(explicit) ? arity == 2 : explicit) ?
        :($nm(X::TreeData) = mapslices((s, c) -> TreeData($into => $nm(s, c)), X; dims=$reduces, coords=true)) :
        :($nm(X::TreeData) = mapslices(s -> TreeData($into => $nm(s)), X; dims=$reduces))
    # a def -> also emit the plain array kernel; a bare name -> just wrap an existing function.
    esc(fdef isa Symbol ? treemethod : Expr(:block, fdef, treemethod))
end
