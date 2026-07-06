using Test
using TreeArrays
using Tables
using FillArrays: Fill

# a small-scale replica of docs/pkpd_demo.jl's chained-quantile shapes -- the
# two acceptance-bar targets for the Tables.jl adapter.
@kernel (:time => :stat) function _tt_compute_stats(L)
    trough, peak = extrema(L)
    baseline = L[1]
    dtrough, dpeak = extrema(L .- baseline)
    (;trough, peak, baseline, dtrough, dpeak)
end

function _tt_stats_percentiles(; n_draws=6, n_subjects=4, n_dense=3, n_cols=5)
    dense_loc(args...) = TreeData(randn(n_draws, n_subjects, n_dense), :draw, :subject, :time=>range(0, 1, n_dense))
    input_draws = TreeData(randn(n_draws, n_cols), :draw, :param; random_effect=:in_sample, placebo=:on, space=:sampler)
    map(Iterators.product(
        TreeDim(:random_effect, (:zero, :population)),
        TreeDim(:placebo, (:on, :off)),
        TreeDim(:schedule, ("some schedule",)),
        TreeDim(:dose, (20, 200)),
    )) do args...
        quantile(
            quantile(_tt_compute_stats(dense_loc(input_draws, args...)), TreeDim(:population, (0.05, 0.5, 0.95)); dims=:subject),
            TreeDim(:posterior, (0.05, 0.5, 0.95)); dims=:draw,
        )
    end
end

function _tt_median_draws(; n_draws=6, n_cols=5)
    input_draws = TreeData(randn(n_draws, n_cols), :draw, :param; random_effect=:in_sample, placebo=:on, space=:sampler)
    quantile(input_draws, TreeDim(:median, 0.5); dims=:draw)
end

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

    @testset "Tables.jl: stats_percentiles schema + melt (stat/population/posterior)" begin
        stats_percentiles = _tt_stats_percentiles()
        @test Tables.istable(typeof(stats_percentiles))

        TreeArrays.MELT_COUNT[] = 0
        sch = Tables.schema(stats_percentiles)
        @test TreeArrays.MELT_COUNT[] == 0   # schema is metadata-only -- never melts
        @test sch.names == (:random_effect, :placebo, :schedule, :dose, :stat, :population, :posterior, :value)
        @test sch.types == (Symbol, Symbol, String, Int, Symbol, Float64, Float64, Float64)
        @test Tables.columnnames(stats_percentiles) == sch.names
        @test TreeArrays.MELT_COUNT[] == 0   # columnnames is metadata-only too

        cols = Tables.columns(stats_percentiles)
        @test TreeArrays.MELT_COUNT[] == 1
        n = 2 * 2 * 1 * 2 * 5 * 3 * 3   # random_effect x placebo x schedule x dose x stat x population x posterior
        for nm in Tables.columnnames(cols)
            col = Tables.getcolumn(cols, nm)
            @test length(col) == n
            @test isconcretetype(eltype(col))   # the "no Any columns" bar -- checked per column,
        end                                     # not via @inferred(getcolumn(::Symbol)) (inherently
        @test TreeArrays.MELT_COUNT[] == 1      # non-monomorphic for a heterogeneous NamedTuple)

        @test Tables.getcolumn(cols, :value) isa Vector{Float64}
        @test Tables.getcolumn(cols, :population) isa Vector{Float64}
        @test Tables.getcolumn(cols, :stat) isa Vector{Symbol}
        @test sort(unique(Tables.getcolumn(cols, :stat))) == sort([:trough, :peak, :baseline, :dtrough, :dpeak])
        @test Tables.getcolumn(cols, :schedule) isa Fill   # constant across every row -- decision oni1bc

        rt = Tables.rowtable(stats_percentiles)
        @test length(rt) == n
        @test Set(keys(rt[1])) == Set(sch.names)
    end

    @testset "Tables.jl: median_draws (scalar quantile -> constant column)" begin
        median_draws = _tt_median_draws(; n_cols=7)
        cols = Tables.columns(median_draws)
        @test Tables.columnnames(cols) == (:random_effect, :placebo, :space, :param, :median, :value)
        @test length(Tables.getcolumn(cols, :value)) == 7
        @test Tables.getcolumn(cols, :param) == 1:7          # unlabelled axis -> 1-based position
        @test Tables.getcolumn(cols, :median) isa Fill        # fixed scalar p -> constant column
        @test all(==(0.5), Tables.getcolumn(cols, :median))
        @test Tables.getcolumn(cols, :random_effect) isa Fill
        @test all(==(:in_sample), Tables.getcolumn(cols, :random_effect))
    end

    @testset "Tables.jl: laziness -- construction never melts" begin
        TreeArrays.MELT_COUNT[] = 0
        stats_percentiles = _tt_stats_percentiles()
        median_draws = _tt_median_draws()
        @test TreeArrays.MELT_COUNT[] == 0   # mapslices/quantile construction touched nothing here
        Tables.columns(stats_percentiles)
        Tables.columns(median_draws)
        @test TreeArrays.MELT_COUNT[] == 2   # exactly one melt per Tables.columns call
    end

    @testset "Tables.jl: unsupported shapes error clearly" begin
        heterogeneous = TreeData(:rec => (;a=TreeData(randn(3), :t), b=5.0))
        @test_throws "heterogeneous records" Tables.schema(heterogeneous)

        rawarray = TreeData(:rec => (;a=TreeData(randn(3), :t), b=[1, 2, 3]))
        @test_throws "carries no dim labels" Tables.schema(rawarray)

        absentfield = TreeData(:rec => (;a=TreeData(randn(3), :t), b=missing))
        @test_throws "absent-dim `missing` sentinel" Tables.schema(absentfield)
    end

end
