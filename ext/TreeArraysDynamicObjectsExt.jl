module TreeArraysDynamicObjectsExt

using TreeArrays
using TreeArrays: meta
using DynamicObjects
using Serialization

# Only structure is serialized here. Every numeric payload, including scalar
# terminals, uses DO's DOMM stream codec and its readonly mapping/validation.
const MAGIC = UInt8[0x54, 0x41, 0x44, 0x4f, 0x4d, 0x4d, 0x41, 0x50] # TADOMMAP
const VERSION = UInt8(1)

struct TreeNode{P,M}
    parent::P
    meta::M
end
struct Payload
    index::Int
end
struct ScalarPayload
    index::Int
end
struct Children{T,N,C}
    size::NTuple{N,Int}
    values::C
end

_describe(x::TreeData, leaves) = TreeNode(_describe(parent(x), leaves), meta(x))
_describe(x::NamedTuple, leaves) = map(v -> _describe(v, leaves), x)
_describe(x::Tuple, leaves) = map(v -> _describe(v, leaves), x)
function _describe(x::AbstractArray{T,N}, leaves) where {T<:TreeData,N}
    values = map(v -> _describe(v, leaves), x)
    Children{T,N,typeof(values)}(size(x), values)
end
function _describe(x::AbstractArray, leaves)
    push!(leaves, x) # references only; never mutate caller-owned storage
    Payload(length(leaves))
end
function _describe(x::Number, leaves)
    push!(leaves, fill(x))
    ScalarPayload(length(leaves))
end
_describe(x, leaves) = throw(ArgumentError(
    "TreeArrays mmap: unsupported terminal $(typeof(x)); use numeric arrays, " *
    "numeric scalars, named/positional records, or arrays of TreeData."))

_restore(x::TreeNode, leaves) = TreeData(_restore(x.parent, leaves), x.meta)
function _payload(index, leaves)
    1 <= index <= length(leaves) || throw(ArgumentError(
        "TreeArrays mmap: structural descriptor references invalid payload $index."))
    leaves[index]
end
_restore(x::Payload, leaves) = _payload(x.index, leaves)
_restore(x::ScalarPayload, leaves) = _payload(x.index, leaves)[]
_restore(x::NamedTuple, leaves) = map(v -> _restore(v, leaves), x)
_restore(x::Tuple, leaves) = map(v -> _restore(v, leaves), x)
function _restore(x::Children{T,N}, leaves) where {T,N}
    # Empty ragged arrays need their element type: there is no representative
    # leaf from which the downstream reduction/Tables type walk can recover it.
    isempty(x.values) && return Array{T,N}(undef, x.size)
    map(v -> _restore(v, leaves), x.values)
end

function _write(io::IO, x::TreeData)
    leaves = AbstractArray[]
    root = _describe(x, leaves)
    metadata = IOBuffer()
    serialize(metadata, (;root, count=length(leaves)))
    bytes = take!(metadata)
    write(io, MAGIC)
    write(io, VERSION)
    write(io, UInt64(length(bytes)))
    write(io, bytes)
    table_start = position(io)
    write(io, zeros(UInt64, length(leaves) + 1))
    offsets = UInt64[]
    for leaf in leaves
        push!(offsets, position(io))
        DynamicObjects.save(Val(:mmap), io, leaf)
    end
    push!(offsets, position(io))
    seek(io, table_start)
    write(io, offsets)
    seek(io, last(offsets))
end

function _read(io::IOStream)
    read(io, length(MAGIC)) == MAGIC || throw(ArgumentError(
        "TreeArrays mmap: invalid container magic."))
    read(io, UInt8) == VERSION || throw(ArgumentError(
        "TreeArrays mmap: unsupported container version."))
    nbytes = read(io, UInt64)
    nbytes <= filesize(io) - position(io) || throw(ArgumentError(
        "TreeArrays mmap: truncated structural metadata."))
    metadata = IOBuffer(read(io, Int(nbytes)))
    description = deserialize(metadata)
    eof(metadata) || throw(ArgumentError("TreeArrays mmap: trailing structural metadata."))
    description.root isa TreeNode || throw(ArgumentError(
        "TreeArrays mmap: the container root must be TreeData."))
    count = description.count
    count isa Int && 0 <= count < (filesize(io) - position(io)) ÷ sizeof(UInt64) || throw(ArgumentError(
        "TreeArrays mmap: invalid numeric payload count."))
    offsets = [read(io, UInt64) for _ in 0:count]
    first(offsets) == position(io) && last(offsets) == filesize(io) &&
        issorted(offsets) || throw(ArgumentError(
            "TreeArrays mmap: invalid numeric payload boundaries or incomplete container."))
    leaves = map(1:count) do index
        seek(io, offsets[index])
        leaf = DynamicObjects.load(Val(:mmap), io; end_offset=offsets[index+1])
        position(io) == offsets[index+1] || throw(ArgumentError(
            "TreeArrays mmap: numeric payload $index does not fill its declared boundary."))
        leaf
    end
    eof(io) || throw(ArgumentError("TreeArrays mmap: trailing numeric payload bytes."))
    _restore(description.root, leaves)
end

function DynamicObjects.save(::Val{:mmap}, path::AbstractString, x::TreeData)
    open(path, "w") do io
        _write(io, x)
    end
    # A zero exit from the writer alone cannot publish an incomplete container.
    # DO supplies an unpublished path; validate after close, before it renames.
    open(_read, path, "r")
    nothing
end

function DynamicObjects.load(::Val{:mmap}, path::AbstractString, ::Type{T}) where {T<:TreeData}
    result = open(_read, path, "r")
    result isa T || throw(ArgumentError(
        "TreeArrays mmap: restored $(typeof(result)) does not satisfy annotation $T; " *
        "annotate TreeData or the compatible array-backed tree type."))
    result
end

function __init__()
    DynamicObjects.register_mmap_container!(MAGIC,
        path -> DynamicObjects.load(Val(:mmap), path, TreeData))
end

end
