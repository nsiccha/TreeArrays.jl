module TreeArrays

using Statistics

include("types.jl")
include("dim_helpers.jl")
include("array_interface.jl")
include("show.jl")
include("mapslices.jl")
include("properties.jl")
include("reducers.jl")
include("kernel.jl")
include("setdim.jl")
include("selectdim.jl")
include("tables.jl")
include("treetable.jl")
include("html.jl")
include("markdown.jl")

export TreeDim, TreeData, TreeArray, TreeNamedTuple, TreeRaggedArray, TreeTuple
export TreeTable
export dims, outerdim
export mapslices, mean, sum, quantile, quantile!
export @kernel
export setdim

end # module TreeArrays
