using Test
using TreeArrays

@testset "TreeArrays" begin

    @testset "sum(X) default + TreeArray array interface" begin
        X = TreeData(randn(4,3), :draw, :param)
        P = parent(X)
        @test sum(X) == sum(P)                # was: UndefKeywordError: dims
        @test size(X) == size(P)              # was: MethodError
        @test length(X) == length(P)          # was: MethodError
        @test ndims(X) == ndims(P)            # was: MethodError
        @test eltype(X) == eltype(P)          # was: Any
        @test X[1] == P[1]                    # was: MethodError (no getindex)
        @test collect(X) == collect(P)        # was: MethodError (no length/iterate)
    end

    # _reduceouter assumes the first ndims(parent(X)) entries of `dims` are exactly the
    # parent array's axes, positionally, in order (_assemble upholds this by construction,
    # but nothing enforces it on a hand-built TreeData). A non-axis dim (here a scalar,
    # `:extra`) placed BEFORE the real axes shifts the positional lookup and silently
    # reduces the wrong parent axis -- confirmed before landing the @assert:
    #   reducing :draw on a well-formed (draw, param) TreeData sums over rows -> [10, 26, 42]
    #   reducing :draw on the misordered (extra, draw, param) TreeData instead summed over
    #   columns -> [15, 18, 21, 24] (wrong axis, wrong shape, silently wrong).
    # Now that _reduceouter asserts the invariant, the misordered construction throws.
    @testset "_reduceouter positional-axis assertion (jz9bkv)" begin
        Pgood = reshape(1.0:12.0, 4, 3)
        Xgood = TreeData(Pgood, :draw, :param)
        Xbad  = TreeData(Pgood, TreeDim(:extra, 99), TreeDim(:draw), TreeDim(:param))
        good  = mapslices(sum, Xgood; dims=:draw)
        @test parent(good) == [10.0, 26.0, 42.0]   # sane baseline: reduces rows, not columns
        threw = try
            mapslices(sum, Xbad; dims=:draw)
            false
        catch e
            e isa AssertionError || rethrow()
            true
        end
        @test threw   # non-axis dim before a real array axis must fail loudly, not mis-map
    end

    # _leafreduce(f, sl::AbstractArray{<:TreeData}) built `vals` via
    # map(i -> ..., eachindex(parent(proto))) -- eachindex is linear, so a multidim leaf's
    # `vals` came back as a flat Vector while the reused `meta(proto)` still described the
    # original multidim shape. Fixed by iterating CartesianIndices(parent(proto)) instead
    # of eachindex, which preserves the leaf's shape.
    @testset "_leafreduce preserves multidim leaf shape (1cn9zad)" begin
        leaves = [TreeData(randn(2, 3), :time, :chan) for _ in 1:5]
        Xouter = TreeData(leaves, :subject)          # TreeRaggedArray: outer :subject axis
        result = mapslices(mean, Xouter; dims=:subject)
        inner  = parent(result)                      # the gathered per-leaf-cell TreeData
        @test size(parent(inner)) == (2, 3)
    end

    @testset "setdim stubs throw instead of silently no-oping" begin
        X = TreeData(randn(4,3), :draw, :param)
        @test (try; setdim(X; foo=:bar); false; catch; true; end)                # was: silently returned X unchanged
        @test (try; setdim((:draw, :param); foo=:bar); false; catch; true; end)  # was: bare error() with no message
    end

    @testset "outer_dim survives TreeData-forwarding" begin
        Xnt = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(4), :draw)))
        Y = TreeData(Xnt, TreeDim(:extra, nothing))
        @test outerdim(Y) === outerdim(Xnt)                    # was: no `outer_dim` field in meta(Y)
        Z = mapslices(mean, Y; dims=:draw)                      # was: errors calling outerdim(Y) inside mapslices
    end

    # The unified quantile no longer coerces p at all -- Base's quantile!(scratch, p)
    # preserves p's container as-is (scalar -> scalar leaf, Vector -> Vector leaf, Tuple ->
    # Tuple leaf), matching decision xxmv6c (option 2). _leafreduce gather-reduces
    # non-array-backed leaves (Tuple- and scalar-parent) directly, so a Tuple/scalar leaf
    # survives a CHAINED quantile call.
    @testset "quantile(X, pdim::TreeDim) leaf shape mirrors pdim's values" begin
        Xq = TreeData(reshape(1.0:12.0, 4, 3), :draw, :param)

        scalar_result = quantile(Xq, TreeDim(:median, 0.5); dims=:draw)
        scalar_leaf = parent(scalar_result)[1]
        @test parent(scalar_leaf) isa Number
        @test !TreeArrays._isaxis(dims(scalar_leaf)[1])       # fixed coordinate, not an axis

        vector_result = quantile(Xq, TreeDim(:pct, [0.25, 0.5, 0.75]); dims=:draw)
        vector_leaf = parent(vector_result)[1]
        @test parent(vector_leaf) isa AbstractVector
        @test TreeArrays._isaxis(dims(vector_leaf)[1])        # real axis

        tuple_result = quantile(Xq, TreeDim(:tup, (0.25, 0.5, 0.75)); dims=:draw)
        tuple_leaf = parent(tuple_result)[1]
        @test TreeArrays.meta(dims(tuple_leaf)[1]).values isa Tuple   # axis label: Tuple stays Tuple
        @test parent(tuple_leaf) isa Tuple

        chained = quantile(tuple_result, TreeDim(:tup2, (0.1, 0.9)); dims=:param)  # was: MethodError CartesianIndices(::Tuple)
        @test parent(parent(chained)) isa Tuple

        chained_scalar = quantile(scalar_result, TreeDim(:median2, 0.5); dims=:param)  # was: MethodError CartesianIndices(::Number)
        @test parent(parent(chained_scalar)) isa Number
    end

end
