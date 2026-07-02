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
_aschild(v::TreeData, inner) = v                 # already a TreeData -> knows its own dims
_aschild(v, inner) = TreeData(v, inner...)       # raw field -> wrap with the inner axes
