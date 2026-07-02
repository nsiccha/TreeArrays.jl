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
