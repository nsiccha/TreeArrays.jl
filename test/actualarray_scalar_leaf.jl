@testitem "TreeActualArray scalar leaves" begin
    using Statistics, NaNStatistics

    function checkscalar(tree, value)
        A = @inferred TreeActualArray(tree)
        @test A isa AbstractArray{typeof(value),0}
        @test size(A) == axes(A) == ()
        @test length(A) == 1
        @test parent(A) === tree
        @test TreeArrays.dims(A) === TreeArrays.dims(tree)
        @test (@inferred A[]) === value
        @test A[1] === A[CartesianIndex()] === value
        @test_throws BoundsError A[2]
        @test_throws BoundsError A[0]
        for copy in (collect(A), Array(A))
            @test copy isa Array{typeof(value),0}
            @test copy[] === value
            copy[] = zero(value)
            @test A[] === value
        end
        @test_throws Base.CanonicalIndexError setindex!(A, zero(value))
        @test A[] === value
    end

    @testset "numeric terminals and metadata" begin
        for value in (3, 2.5f0, 4.0, 1//3, 2 + 3im, true)
            checkscalar(TreeData(value), value)
            checkscalar(TreeData(value, TreeDim(:fixed, :label), TreeDim(:draw, nothing)), value)
        end
        # A physical axis cannot silently vanish from a scalar's array view.
        @test_throws "scalar-backed leaf cannot carry real axes" TreeActualArray(TreeData(2.0, :draw))
        @test_throws "scalar-backed leaf cannot carry real axes" TreeActualArray(TreeData(2.0, :draw => (1,)))
    end

    @testset "fully reduced numeric results" begin
        values = reshape(collect(1.0:4000.0), 1000, 4)
        original = copy(values)
        tree = TreeData(values, :draw, :chain)
        med = quantile(tree, TreeDim(:median, 0.5); dims=(:draw, :chain))
        checkscalar(med, quantile(vec(values), 0.5))
        @test length(TreeArrays.dims(med)) == 3
        for reducer in (mean, sum, std, var)
            checkscalar(reducer(tree; dims=(:draw, :chain)), reducer(values))
        end
        record = quantile(tree, (q25=0.25, q50=0.5); dims=(:draw, :chain))
        checkscalar(record.q25, quantile(vec(values), 0.25))
        checkscalar(record.q50, quantile(vec(values), 0.5))
        oneaxis = quantile(TreeData(collect(1.0:100.0), :draw), (q50=0.5,); dims=:draw)
        checkscalar(oneaxis.q50, 50.5)
        withnan = TreeData([1.0, NaN, 3.0], :draw)
        checkscalar(nanquantile(withnan, TreeDim(:median, 0.5); dims=:draw), 2.0)
        checkscalar(nanquantile(withnan, (q50=0.5,); dims=:draw).q50, 2.0)
        @test Base.get_extension(TreeArrays, :TreeArraysNaNStatisticsExt) !== nothing
        @test values == original
    end

    @testset "wrapped scalar fields use the same recursive walk" begin
        record = TreeData(:stat => (; a=TreeData(2.0), b=TreeData(3.0)))
        A = TreeActualArray(record)
        @test size(A) == (2,)
        @test collect(A) == [2.0, 3.0]
        @test TreeActualArray(TreeData(fill(4.0)))[] === 4.0
        @test TreeActualArray(TreeData(fill(TreeData(5.0))))[] === 5.0
    end
end
