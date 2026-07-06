using Test
using TreeArrays
using Tables
using NaNStatistics
using Statistics

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

    # quantile(X, p::NamedTuple; dims) -- the Tables wide-emit's producer side
    # (Delta B): same one-pass quantile! as the TreeDim form, packed into a
    # TreeNamedTuple record leaf (fields named by keys(p)) instead of a band axis.
    @testset "quantile(X, p::NamedTuple; dims) -> TreeNamedTuple record leaf" begin
        Xq = TreeData(reshape(1.0:12.0, 4, 3), :draw, :param)
        p = (q025=0.25, median=0.5, q975=0.75)

        r = quantile(Xq, p; dims=:draw)
        leaf = first(parent(r))
        @test leaf isa TreeArrays.TreeNamedTuple
        @test TreeArrays.name(TreeArrays.outerdim(leaf)) === :quantile   # cosmetic/show-only default
        @test parent(leaf) == NamedTuple{keys(p)}(Statistics.quantile(1.0:4.0, values(p)))

        # every column position matches direct Statistics.quantile on the raw slice
        for j in 1:3
            leaf_j = parent(r)[j]
            @test parent(leaf_j) == NamedTuple{keys(p)}(Statistics.quantile(Float64.(4j-3:4j), values(p)))
        end

        # wide-emit Tables machinery (Delta A): the record fans out to one
        # column per quantile level, name-prefixed by nothing (bare field
        # name, since each is a terminal :value) -- no field-key column.
        cols = Tables.columns(r)
        @test Set(Tables.columnnames(cols)) == Set((:param, keys(p)...))
        for k in keys(p)
            @test length(Tables.getcolumn(cols, k)) == 3
            @test isconcretetype(eltype(Tables.getcolumn(cols, k)))
        end
        rt = Tables.rowtable(r)
        @test length(rt) == 3
        for row in rt
            j = row.param
            expected = NamedTuple{keys(p)}(Statistics.quantile(Float64.(4j-3:4j), values(p)))
            @test all(k -> row[k] == expected[k], keys(p))
        end
    end

    # NaN-aware quantile lives in a package extension (todo b1am3w), not a
    # `skipnan` kwarg on Statistics.quantile (user override of 1qbk7u4) --
    # Statistics.quantile itself is untouched and keeps throwing on NaN.
    @testset "NaNStatistics.nanquantile(X, pdim::TreeDim; dims) extension" begin
        ext = Base.get_extension(TreeArrays, :TreeArraysNaNStatisticsExt)
        @test ext !== nothing

        Xq = TreeData(reshape(1.0:12.0, 4, 3), :draw, :param)
        pdim = TreeDim(:pct, (0.25, 0.5, 0.75))

        @testset "no-NaN data matches Statistics.quantile exactly" begin
            @test parent(nanquantile(Xq, pdim; dims=:draw)) == parent(quantile(Xq, pdim; dims=:draw))
        end

        @testset "some-NaN slice: quantiles over the survivors; Statistics.quantile still throws" begin
            P = Array(reshape(1.0:12.0, 4, 3))
            P[1, 1] = NaN
            Xn = TreeData(P, :draw, :param)
            @test_throws Exception quantile(Xn, pdim; dims=:draw)

            r = nanquantile(Xn, pdim; dims=:draw)
            @test parent(parent(r)[1]) == Statistics.quantile(filter(!isnan, P[:, 1]), (0.25, 0.5, 0.75))
            @test parent(parent(r)[2]) == Statistics.quantile(P[:, 2], (0.25, 0.5, 0.75))   # untouched column unaffected
        end

        @testset "all-NaN slice: NaN at every level, container shape mirrored, never throws" begin
            P = Array(reshape(1.0:12.0, 4, 3))
            P[:, 1] .= NaN
            Xn = TreeData(P, :draw, :param)

            r_tup = nanquantile(Xn, TreeDim(:pct, (0.25, 0.5, 0.75)); dims=:draw)
            @test all(isnan, parent(parent(r_tup)[1]))
            @test parent(parent(r_tup)[1]) isa NTuple{3,Float64}

            r_scalar = nanquantile(Xn, TreeDim(:median, 0.5); dims=:draw)
            @test isnan(parent(parent(r_scalar)[1]))

            r_vec = nanquantile(Xn, TreeDim(:pct2, [0.25, 0.5, 0.75]); dims=:draw)
            @test all(isnan, parent(parent(r_vec)[1]))
            @test parent(parent(r_vec)[1]) isa Vector{Float64}
        end

        @testset "type-stability: all-NaN leaf type matches a normal slice's leaf type (Float32 data)" begin
            P32 = Float32.(reshape(1:12, 4, 3))
            P32[:, 1] .= NaN32
            X32n = TreeData(P32, :draw, :param)
            r = nanquantile(X32n, pdim; dims=:draw)
            @test typeof(parent(parent(r)[1])) == typeof(parent(parent(r)[2]))   # all-NaN vs normal leaf
            @test isconcretetype(eltype(parent(r)))                              # the gathered TreeData is concrete
        end

        @testset "no per-slice allocation regression (shared scratch, O(1) per slice)" begin
            n_draws, n_cols = 50, 200
            X = TreeData(randn(n_draws, n_cols), :draw, :param)
            pdim3 = TreeDim(:pct, (0.1, 0.5, 0.9))
            quantile(X, pdim3; dims=:draw)          # warm up both paths
            nanquantile(X, pdim3; dims=:draw)
            a_base = @allocated quantile(X, pdim3; dims=:draw)
            a_nan  = @allocated nanquantile(X, pdim3; dims=:draw)
            @test a_nan == a_base   # byte-identical per-slice profile to the untouched Statistics.quantile baseline

            Xbig = TreeData(randn(n_draws, 100 * n_cols), :draw, :param)
            nanquantile(Xbig, pdim3; dims=:draw)
            a_big = @allocated nanquantile(Xbig, pdim3; dims=:draw)
            # O(1) per slice, not O(nslices): a real per-slice leak (as found and
            # fixed mid-implementation) roughly TRIPLED the per-slice allocation at
            # this scale gap -- isapprox (not exact ==) absorbs fixed per-call noise.
            @test isapprox(a_big / (100 * n_cols), a_nan / n_cols; rtol=0.3)
        end

        @testset "NamedTuple p -> TreeNamedTuple record leaf (mirrors quantile(X, p::NamedTuple; dims))" begin
            p = (q025=0.25, median=0.5, q975=0.75)

            @testset "no-NaN data matches quantile(X, p::NamedTuple; dims) exactly" begin
                @test parent(first(parent(nanquantile(Xq, p; dims=:draw)))) ==
                      parent(first(parent(quantile(Xq, p; dims=:draw))))
            end

            @testset "some-NaN slice: quantiles over the survivors" begin
                P = Array(reshape(1.0:12.0, 4, 3))
                P[1, 1] = NaN
                Xn = TreeData(P, :draw, :param)
                r = nanquantile(Xn, p; dims=:draw)
                @test parent(first(parent(r))) == NamedTuple{keys(p)}(Statistics.quantile(filter(!isnan, P[:, 1]), values(p)))
            end

            @testset "all-NaN slice: NaN at every field, never throws" begin
                P = Array(reshape(1.0:12.0, 4, 3))
                P[:, 1] .= NaN
                Xn = TreeData(P, :draw, :param)
                r = nanquantile(Xn, p; dims=:draw)
                leaf = parent(first(parent(r)))
                @test all(isnan, values(leaf))
                @test leaf isa NamedTuple{keys(p),NTuple{3,Float64}}
            end
        end
    end

    @testset "Tables.jl: stats_percentiles schema + view columns (stat/population/posterior)" begin
        stats_percentiles = _tt_stats_percentiles()
        @test Tables.istable(typeof(stats_percentiles))

        # wide-emit (decision 1kpyu7n): the `:stat` record (trough/peak/
        # baseline/dtrough/dpeak) fans out to one column per field -- no
        # field-key column, no row-dim of its own. All 5 fields share the
        # SAME population/posterior axes (validated by `_fieldsig`), so
        # those are de-duplicated to ONE column pair, not 5.
        stat_fields = (:trough, :peak, :baseline, :dtrough, :dpeak)
        sch = Tables.schema(stats_percentiles)
        @test Set(sch.names) == Set((:random_effect, :placebo, :schedule, :dose, :population, :posterior, stat_fields...))
        @test Tables.columnnames(stats_percentiles) == sch.names

        cols = Tables.columns(stats_percentiles)
        @test Tables.columnnames(cols) == sch.names   # schema and the actual columns always agree (both mirror the same walk)
        n = 2 * 2 * 1 * 2 * 3 * 3   # random_effect x placebo x schedule x dose x population x posterior (stat contributes no row-dim)
        for nm in Tables.columnnames(cols)
            col = Tables.getcolumn(cols, nm)
            @test length(col) == n
            @test isconcretetype(eltype(col))   # the "no Any columns" bar -- checked per column,
        end                                     # not via @inferred(getcolumn(::Symbol)) (inherently
                                                 # non-monomorphic for a heterogeneous NamedTuple)

        for f in stat_fields
            @test Tables.getcolumn(cols, f) isa TreeArrays.ValueColumn{Float64}
        end
        @test Tables.getcolumn(cols, :population) isa TreeArrays.AxisColumn{Float64}
        @test Tables.getcolumn(cols, :schedule) isa TreeArrays.AxisColumn{String}   # a real (length-1) axis, swept via Iterators.product -- decision oni1bc
        @test all(==("some schedule"), Tables.getcolumn(cols, :schedule))

        rt = Tables.rowtable(stats_percentiles)
        @test length(rt) == n
        @test Set(keys(rt[1])) == Set(sch.names)

        # value correctness: every stat field's value is finite (a real, if
        # partial, cross-check against the underlying computation -- exact
        # reproduction of the nested-quantile pipeline is covered by the
        # dedicated hand-built reference test below).
        for f in stat_fields
            @test all(isfinite, collect(Tables.getcolumn(cols, f)))
        end
    end

    @testset "Tables.jl: median_draws (scalar quantile -> constant column)" begin
        median_draws = _tt_median_draws(; n_cols=7)
        sch = Tables.schema(median_draws)
        cols = Tables.columns(median_draws)
        @test Tables.columnnames(cols) == sch.names   # always consistent now (both mirror _schema)
        @test length(Tables.getcolumn(cols, :value)) == 7
        @test Tables.getcolumn(cols, :param) == 1:7          # unlabelled axis -> 1-based position
        @test Tables.getcolumn(cols, :median) isa TreeArrays.ConstColumn{Float64}   # fixed scalar p -> constant column
        @test all(==(0.5), Tables.getcolumn(cols, :median))
        @test Tables.getcolumn(cols, :random_effect) isa TreeArrays.ConstColumn{Symbol}
        @test all(==(:in_sample), Tables.getcolumn(cols, :random_effect))
    end

    @testset "Tables.jl: view columns -- value correctness against a hand-computed reference" begin
        # small, fully deterministic 2-level tree: outer labelled :scenario axis (3
        # positions) x inner (:draw, :param) leaf matrix -- every value known exactly,
        # independent of any TreeArrays internals (a direct arithmetic formula).
        leafvalue(s, d, p) = 100.0 * s + 10.0 * d + p
        leaves = [TreeData([leafvalue(s, d, p) for d in 1:4, p in 1:3], :draw, :param) for s in 1:3]
        X = TreeData(leaves, TreeDim(:scenario, (:a, :b, :c)))
        symfor = Dict(1 => :a, 2 => :b, 3 => :c)

        rt = Tables.rowtable(X)
        @test length(rt) == 3 * 4 * 3
        expected = Set(
            (scenario=symfor[s], draw=d, param=p, value=leafvalue(s, d, p))
            for s in 1:3, d in 1:4, p in 1:3
        )
        @test Set(rt) == expected   # order-agnostic per item 2 of the design (row order is a fresh choice, immaterial to AoV)

        # a NamedTuple-record leaf on top (wide-emit -- decision 1kpyu7n): no
        # record-key column, no row-dim of its own -- fields `a`/`b` fan out
        # to their own terminal columns, but SHARE their common fixed :tag
        # dim as ONE column (both fields agree on it, validated by
        # `_fieldsig` -- scope-fork 2), not a duplicated a_tag/b_tag pair.
        recvalue(s, which) = which === :a ? 1000.0 + s : 2000.0 + s
        recleaves = [TreeData(:rec => (; a=TreeData(recvalue(s, :a), TreeDim(:tag, :fixedtag)), b=TreeData(recvalue(s, :b), TreeDim(:tag, :fixedtag)))) for s in 1:3]
        Y = TreeData(recleaves, TreeDim(:scenario, (:a, :b, :c)))
        rty = Tables.rowtable(Y)
        @test length(rty) == 3
        expectedy = Set(
            (scenario=symfor[s], tag=:fixedtag, a=recvalue(s, :a), b=recvalue(s, :b))
            for s in 1:3
        )
        @test Set(rty) == expectedy
    end

    @testset "Tables.jl: laziness -- construction never densifies, O(1) per access" begin
        # construction: flat in row count (no per-row work at Tables.columns time)
        small = _tt_median_draws(; n_cols=50)
        Tables.columns(small)   # warm
        a_small = @allocated Tables.columns(small)

        big = _tt_median_draws(; n_cols=5000)   # 100x more rows via a plain axis (no ragged-check boundary)
        Tables.columns(big)
        a_big = @allocated Tables.columns(big)
        @test a_big <= a_small * 4   # not proportional to the 100x row-count growth -- well below what a real per-row leak would show

        # access: bulk enumeration over a column is O(1) allocation, not O(N) --
        # this is the Bruno enumeration hot path (rowtable-style iteration).
        median_small = _tt_median_draws(; n_cols=50)
        median_big = _tt_median_draws(; n_cols=5000)
        cols_small = Tables.columns(median_small)
        cols_big = Tables.columns(median_big)
        function bulk_access(col)
            s = 0
            for i in eachindex(col)
                s += hash(col[i])
            end
            s
        end
        for nm in Tables.columnnames(cols_small)
            col_s, col_b = Tables.getcolumn(cols_small, nm), Tables.getcolumn(cols_big, nm)
            bulk_access(col_s); bulk_access(col_b)   # warm
            a_s, a_b = (@allocated bulk_access(col_s)), (@allocated bulk_access(col_b))
            @test a_b <= max(a_s, 64) * 2   # flat regardless of the 100x row-count gap
        end

        # storage: ConstColumn/AxisColumn stay O(depth)/O(axis-length), never O(rows)
        n_cols = 5000
        big_median = _tt_median_draws(; n_cols)
        bcols = Tables.columns(big_median)
        const_col = Tables.getcolumn(bcols, :median)
        axis_col = Tables.getcolumn(bcols, :param)
        @test const_col isa TreeArrays.ConstColumn
        @test Base.summarysize(const_col) < Base.summarysize(collect(const_col))
        @test Base.summarysize(axis_col) < Base.summarysize(collect(axis_col))
        value_col = Tables.getcolumn(bcols, :value)
        @test value_col isa TreeArrays.ValueColumn
        @test value_col.x === big_median   # genuinely a reference, never a copy
    end

    @testset "Tables.jl: acceptance gate -- @allocated flat, homogeneous AND heterogeneous records" begin
        # the hard acceptance gate (decision 1vbt15w / user: "I just hope
        # there's no actually allocated vector anywhere"): Tables.columns
        # assembles lazy view columns only, for BOTH a homogeneous-type
        # record and a heterogeneous-type one (different fields, different
        # concrete eltypes, sharing a fixed dim) -- allocation must stay
        # flat as the outer row-count scales, not grow proportionally.
        _tt_hetero_record(; n_cols) =
            TreeData([TreeData(:rec => (; a=TreeData(1.0 * j, TreeDim(:tag, :x)), b=TreeData(j, TreeDim(:tag, :x)))) for j in 1:n_cols], :param)

        homog_small = _tt_median_draws(; n_cols=50)
        homog_big = _tt_median_draws(; n_cols=5000)
        Tables.columns(homog_small); Tables.columns(homog_big)   # warm
        a_homog_small = @allocated Tables.columns(homog_small)
        a_homog_big = @allocated Tables.columns(homog_big)
        @test a_homog_big <= a_homog_small * 4

        hetero_small = _tt_hetero_record(; n_cols=50)
        hetero_big = _tt_hetero_record(; n_cols=5000)
        Tables.columns(hetero_small); Tables.columns(hetero_big)   # warm
        a_hetero_small = @allocated Tables.columns(hetero_small)
        a_hetero_big = @allocated Tables.columns(hetero_big)
        @test a_hetero_big <= a_hetero_small * 4

        hcols = Tables.columns(hetero_big)
        @test Tables.getcolumn(hcols, :a) isa TreeArrays.ValueColumn{Float64}   # each heterogeneous field: its own
        @test Tables.getcolumn(hcols, :b) isa TreeArrays.ValueColumn{Int}       # concretely-typed view, no Union/box
        @test Tables.getcolumn(hcols, :tag) isa TreeArrays.ConstColumn{Symbol}  # shared fixed dim, de-duplicated

        @info "Delta A acceptance gate: Tables.columns(X) @allocated (50 vs 5000 cols)" a_homog_small a_homog_big a_hetero_small a_hetero_big
    end

    @testset "Tables.jl: ragged trees error clearly at Tables.columns (schema still succeeds -- type-only)" begin
        # matches docs/pkpd_demo.jl's input_data shape: per-subject arrays of
        # genuinely differing length under an outer :subject axis.
        n_subjects = 4
        n_measurements = [3, 5, 3, 3]   # subject 2 differs -- ragged
        measurement = map(n -> TreeData(randn(n), :time => sort(randn(n))), n_measurements)
        ragged = TreeData(measurement, :subject)

        @test Tables.schema(ragged) isa Tables.Schema   # type-only -- physically cannot see instance raggedness
        @test_throws "ragged trees are not a supported Tables shape yet" Tables.columns(ragged)

        # the regular (same length, identical coordinates) counterpart works fine
        shared_times = sort(randn(3))
        regular = TreeData(map(_ -> TreeData(randn(3), :time => shared_times), 1:n_subjects), :subject)
        cols = Tables.columns(regular)
        @test length(Tables.getcolumn(cols, :value)) == n_subjects * 3
    end

    @testset "Tables.jl: coordinate-value guard (===-fast-path + isequal fallback, Option A)" begin
        # false positive 1: siblings share the SAME coordinate object -- every
        # comparison is an `===` hit, O(1), zero `isequal` calls needed.
        shared_times = sort(randn(3))
        shared_obj = TreeData(map(_ -> TreeData(randn(3), :time => shared_times), 1:4), :subject)
        @test length(Tables.getcolumn(Tables.columns(shared_obj), :value)) == 12

        # false positive 2: siblings each get their OWN, independently-built but
        # value-equal coordinate array -- `===` misses, `isequal` accepts, still
        # NOT rejected as ragged.
        distinct_but_equal = TreeData(map(_ -> TreeData(randn(3), :time => copy(shared_times)), 1:4), :subject)
        @test length(Tables.getcolumn(Tables.columns(distinct_but_equal), :value)) == 12

        # true positive, single-level: same lengths, genuinely different
        # per-sibling coordinate VALUES -- rejected, message names the dim.
        differing = TreeData([TreeData(randn(3), :time => sort(randn(3))) for _ in 1:4], :subject)
        @test_throws "ragged trees are not a supported Tables shape yet" Tables.columns(differing)
        @test_throws "time" Tables.columns(differing)
        @test Tables.schema(differing) isa Tables.Schema   # schema still succeeds -- type-only

        # true positive, multi-level: TWO top-level siblings, each internally
        # uniform (all of subject A's visits share ONE :time array, all of
        # subject B's visits share a DIFFERENT one) -- an own-level-only check
        # would miss this (each boundary's own siblings look consistent); the
        # full recursive coordinate signature catches it via the representative
        # (first-visit) path comparison at the TOP boundary.
        time_a, time_b = sort(randn(2)), sort(randn(2))
        subject_a = TreeData([TreeData(randn(2), :time => time_a) for _ in 1:3], :visit)
        subject_b = TreeData([TreeData(randn(2), :time => time_b) for _ in 1:3], :visit)
        nested_differing = TreeData([subject_a, subject_b], :subject)
        @test_throws "ragged trees are not a supported Tables shape yet" Tables.columns(nested_differing)
        @test_throws "time" Tables.columns(nested_differing)

        # the multi-level REGULAR counterpart (both subjects share ONE :time
        # object across all visits, all subjects) still works.
        time_shared = sort(randn(2))
        subject_a2 = TreeData([TreeData(randn(2), :time => time_shared) for _ in 1:3], :visit)
        subject_b2 = TreeData([TreeData(randn(2), :time => time_shared) for _ in 1:3], :visit)
        nested_regular = TreeData([subject_a2, subject_b2], :subject)
        @test length(Tables.getcolumn(Tables.columns(nested_regular), :value)) == 2 * 3 * 2
    end

    @testset "Tables.jl: unsupported shapes error clearly" begin
        # differing ROWDIMS across fields (here: `a` carries a real :t axis,
        # `b` is a bare scalar) is scope-fork-3 -- a confirmed non-goal, same
        # footing as any other ragged shape. `Tables.schema` is type-only and
        # can't see rowdims (an instance-only fact), so it still succeeds;
        # `Tables.columns` is where this is actually caught.
        rowdims_mismatch = TreeData(:rec => (;a=TreeData(randn(3), :t), b=5.0))
        @test Tables.schema(rowdims_mismatch) isa Tables.Schema
        @test_throws "inconsistent shape" Tables.columns(rowdims_mismatch)

        # genuine field-TYPE heterogeneity (decision 1vbt15w, relax) -- SAME
        # rowdims (both fields are scalar, fixed-dim-only leaves), different
        # VALUE types (Float64 vs Int) -- fully supported: schema and
        # columns agree, each field gets its own concretely-typed column.
        type_hetero = TreeData(:rec => (;a=TreeData(1.0, TreeDim(:tag, :x)), b=TreeData(2, TreeDim(:tag, :x))))
        sch = Tables.schema(type_hetero)
        cols = Tables.columns(type_hetero)
        @test Tables.columnnames(cols) == sch.names
        @test Tables.getcolumn(cols, :a) isa TreeArrays.ValueColumn{Float64}
        @test Tables.getcolumn(cols, :b) isa TreeArrays.ValueColumn{Int}

        rawarray = TreeData(:rec => (;a=TreeData(randn(3), :t), b=[1, 2, 3]))
        @test_throws "carries no dim labels" Tables.schema(rawarray)

        absentfield = TreeData(:rec => (;a=TreeData(randn(3), :t), b=missing))
        @test_throws "absent-dim `missing` sentinel" Tables.schema(absentfield)
    end

end
