@testitem "TreeActualArray scalar grids" begin
    using Test
    using TreeArrays
    using NaNStatistics
    using Statistics

    # Public AbstractArray protocol control: preserve actual outer axes,
    # including an array whose coordinate positions do not start at one.
    struct ShiftedScalarGrid{T} <: AbstractMatrix{T}
        storage::Matrix{T}
    end
    Base.size(X::ShiftedScalarGrid) = size(X.storage)
    Base.axes(::ShiftedScalarGrid) = (Base.IdentityUnitRange(-1:0), Base.IdentityUnitRange(2:4))
    Base.IndexStyle(::Type{<:ShiftedScalarGrid}) = IndexCartesian()
    function Base.getindex(X::ShiftedScalarGrid, i::Int, j::Int)
        @boundscheck checkbounds(X, i, j)
        X.storage[i + 2, j - 1]
    end

    @testset "outer shape, order, coordinates and ordinary array indexing" begin
        for shape in ((5,), (2, 3), (2, 1, 3), (1, 2, 1, 2))
            values = reshape(collect(1.0:prod(shape)), shape)
            leaves = map(v -> TreeData(v, TreeDim(:subject, nothing)), values)
            ds = ntuple(j -> TreeDim(Symbol(:axis, j), collect(10:10:10 * shape[j])), length(shape))
            tree = TreeData(leaves, ds..., TreeDim(:dose, 20))
            A = TreeActualArray(tree)
            @test A isa AbstractArray{Float64,length(shape)}
            @test size(A) == shape
            @test axes(A) == axes(leaves)
            @test parent(A) === tree
            @test dims(A) === dims(tree)
            @test collect(A) == values
            @test vec(collect(A)) == vec(values)
            @test sum(A) == sum(values)
            @test A[:] == vec(values)
            for j in eachindex(shape)
                @test coords(dims(A)[j]) === coords(ds[j])
            end
            for I in CartesianIndices(values)
                @test (@inferred A[I]) === values[I]
                @test (@inferred getindex(A, Tuple(I)...)) === values[I]
            end
            for i in eachindex(values)
                @test A[i] === values[i]
            end
            @test A[ntuple(_ -> 1, length(shape))..., 1] === first(values)
            @test_throws BoundsError A[0]
            @test_throws BoundsError A[length(A) + 1]
            @test_throws BoundsError A[CartesianIndex(ntuple(j -> j == 1 ? 0 : 1, length(shape)))]
            @test_throws BoundsError A[ntuple(_ -> 1, length(shape))..., 2]
            @test_throws Base.CanonicalIndexError setindex!(A, 0.0, ntuple(_ -> 1, length(shape))...)
        end
    end

    @testset "numeric types, singleton, zero rank and typed empty grids" begin
        for T in (Int, Float32, Float64, ComplexF64, Rational{Int}, BigFloat, Bool)
            v = T === Bool ? [false, true] : T[1, 2]
            A = TreeActualArray(TreeData(map(TreeData, v), :position))
            @test eltype(A) === T
            @test collect(A) == v
            @test (@inferred A[2]) === v[2]
        end
        singleton = TreeActualArray(TreeData([TreeData(3.0)], :position => (:only,)))
        @test size(singleton) == (1,)
        @test singleton[1] === 3.0
        @test coords(dims(singleton)[1]) === (:only,)
        makeview(tree) = TreeActualArray(tree)
        @test (@inferred makeview(TreeData([TreeData(3.0)], :position))) isa AbstractVector{Float64}
        scalar = TreeActualArray(TreeData(fill(TreeData(4.0))))
        @test size(scalar) == axes(scalar) == ()
        @test scalar[] === scalar[1] === scalar[CartesianIndex()] === 4.0
        @test_throws BoundsError scalar[2]

        proto = TreeData(0.0, TreeDim(:subject, nothing), TreeDim(:dose, 20))
        for shape in ((0,), (0, 2), (2, 0, 1))
            leaves = Array{typeof(proto)}(undef, shape)
            tree = TreeData(leaves, ntuple(j -> TreeDim(Symbol(:axis, j), 1:shape[j]), length(shape))...)
            A = TreeActualArray(tree)
            @test eltype(A) === Float64
            @test size(A) == shape
            @test axes(A) == axes(leaves)
            @test isempty(A)
            @test collect(A) == zeros(shape)
            @test sum(A) == 0.0
            @test_throws BoundsError A[1]
        end
        broad = TreeData[TreeData(1.0), TreeData(2.0)]
        @test eltype(TreeActualArray(TreeData(broad, :position))) === Float64
        @test_throws ErrorException TreeActualArray(TreeData(TreeData[], :position))
        @test_throws "concrete numeric scalar" TreeActualArray(TreeData(TreeData{Float64}[], :position))
    end

    @testset "outer axes are preserved" begin
        shifted = ShiftedScalarGrid(reshape([TreeData(Float64(i)) for i in 1:6], 2, 3))
        tree = TreeData(shifted, :row => (:a, :b), :column => (20, 30, 40))
        A = TreeActualArray(tree)
        @test axes(A) == axes(shifted)
        @test size(A) == (2, 3)
        @test A[-1, 2] === 1.0
        @test A[0, 4] === 6.0
        @test [A[i] for i in 1:6] == collect(1.0:6.0)
        @test Array(A) == reshape(collect(1.0:6.0), 2, 3)
        @test collect(A) == reshape(collect(1.0:6.0), 2, 3)
        for (i, I) in enumerate(CartesianIndices(A))
            @test A[I] === Float64(i)
        end
        @test coords(dims(A)[1]) === (:a, :b)
        @test_throws BoundsError A[1, 2]
        @test_throws BoundsError A[-1, 1]
        @test_throws BoundsError A[0]
    end

    @testset "live alias, subarray backing and lifetime" begin
        leaves = reshape([TreeData(Float64(i)) for i in 1:12], 3, 4)
        tree = TreeData(view(leaves, 1:2, 2:4), :row, :column)
        A = TreeActualArray(tree)
        @test parent(parent(A)) === parent(tree)
        @test collect(A) == [4.0 7.0 10.0; 5.0 8.0 11.0]
        leaves[2, 3] = TreeData(-77.0)
        @test A[2, 2] === -77.0            # a packed snapshot would fail here
        @test_throws Base.CanonicalIndexError setindex!(A, 5.0, 2, 2)
        @test parent(leaves[2, 3]) === -77.0
        retained, weak = let storage = [TreeData(7.0)]
            (TreeActualArray(TreeData(storage, :position)), WeakRef(storage))
        end
        GC.gc()
        @test weak.value !== nothing
        @test parent(parent(retained)) === weak.value
        @test retained[1] === 7.0
    end

    @testset "actual N-D nanquantile field extraction" begin
        @test Base.get_extension(TreeArrays, :TreeArraysNaNStatisticsExt) !== nothing
        values = reshape(collect(1.0:120.0), 5, 3, 2, 4)
        values[2, 2, 1, 3] = NaN
        values[:, 3, 2, 4] .= NaN
        X = TreeData(values, :subject, :time => (0.0, 1.0, 4.0),
                     :qoi => (:a, :b), :draw => 101:104)
        field = nanquantile(X, (p05=0.05, p50=0.5, p95=0.95); dims=:subject).p05
        A = TreeActualArray(field)
        @test A isa AbstractArray{Float64,3}
        @test size(A) == (3, 2, 4)
        @test axes(A) == axes(parent(field))
        @test parent(A) === field
        @test map(TreeArrays.name, dims(A)) == (:time, :qoi, :draw, :subject)
        @test coords(dims(A)[1]) === (0.0, 1.0, 4.0)
        @test coords(dims(A)[2]) === (:a, :b)
        @test coords(dims(A)[3]) == 101:104
        for I in CartesianIndices(A)
            sample = filter(!isnan, values[:, Tuple(I)...])
            expected = isempty(sample) ? NaN : quantile(sample, 0.05)
            @test isequal(A[I], expected)
            @test isequal(A[LinearIndices(A)[I]], expected)
        end
        @test isequal(collect(A), map(leaf -> parent(leaf), parent(field)))
        @test_throws BoundsError A[4, 1, 1]
    end

    @testset "unsupported terminal representations and keywords" begin
        @test_throws "same numeric scalar type" TreeActualArray(TreeData(TreeData[TreeData(1), TreeData(2.0)], :position))
        @test_throws "numeric scalar terminal" TreeActualArray(TreeData([TreeData("one"), TreeData("two")], :position))
        mixed = TreeData(TreeData[TreeData(1.0), TreeData(fill(2.0))], :position)
        @test_throws "zero-dimensional array-backed" TreeActualArray(mixed)
        live = TreeData([TreeData(1.0, :inner), TreeData(2.0, :inner)], :position)
        @test_throws "real inner axes" TreeActualArray(live)
        @test_throws "one dimension label" TreeActualArray(TreeData(fill(TreeData(1.0), 2, 3), :position))
        @test_throws "coordinates for" TreeActualArray(TreeData([TreeData(1.0), TreeData(2.0)], :position => (:one,)))

        # 0D array leaves stay on the established array-subtree path, even
        # though their shape signatures match dimensionless scalar terminals.
        zero_array = TreeData(fill(9.0))
        @test TreeActualArray(zero_array)[] === 9.0     # existing dense 0D support
        @test_throws "even when rectangular" TreeActualArray(TreeData([zero_array, zero_array], :position))
        @test_throws "even when rectangular" TreeActualArray(TreeData(typeof(zero_array)[], :position))
        nested = TreeData([TreeData([TreeData(1.0)], :inner)], :outer)
        @test_throws "nested containers are unsupported" TreeActualArray(nested)
        grid = TreeData([TreeData(1.0), TreeData(2.0)], :position)
        @test_throws "`align=` naming the policy" TreeActualArray(grid; dims=:position)
        @test_throws "`dims=` naming the ragged inner axis" TreeActualArray(grid; align=:error_equal)
        @test_throws "array-backed" TreeActualArray(grid; dims=:position, align=:error_equal)
    end
end
