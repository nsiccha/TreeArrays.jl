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
