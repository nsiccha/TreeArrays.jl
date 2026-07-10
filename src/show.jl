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

print_values(io::IO, X::TreeData) = begin
    get(io, :compact, false) || print(io, "\n")
    print(_limitctx(io), parent(X))
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
Base.show(io::IO, D::TreeDim) = begin
    print(io, name(D), ": ")
    print(IOContext(io, :compact => true), meta(D).values)
end
