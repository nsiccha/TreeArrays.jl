module TreeArrays

using Statistics

include("types.jl")
include("dim_helpers.jl")
include("array_interface.jl")
include("show.jl")
include("mapslices.jl")
include("reducers.jl")
include("kernel.jl")
include("setdim.jl")
include("tables.jl")

export TreeDim, TreeData, TreeArray, TreeNamedTuple, TreeRaggedArray, TreeTuple
export dims, outerdim
export mapslices, mean, sum, quantile, quantile!
export @kernel
export setdim

end # module TreeArrays
