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

# the coords opt-in, in every `@kernel` shape (snag kernels-cannot-s). Top level, because a
# macro-emitted `function` belongs where the methods it defines are visible to every testset.
_tt_trapz(t, y) = sum(i -> (t[i+1] - t[i]) * (y[i+1] + y[i]) / 2, 1:length(t)-1)
# TWO plain positional args -> the coordinate vector arrives as the second one.
@kernel (:time => :stat) function _tt_nca(y, t)
    (; cmax = maximum(y), tmax = t[argmax(y)], auc = _tt_trapz(t, y))
end
# a DEFAULT / a SPLAT: still 1-slice kernels, NOT coordinate consumers.
@kernel (:time => :stat) function _tt_withdefault(L, scale = 2)
    (; peak = maximum(L) * scale)
end
@kernel (:time => :stat) function _tt_withsplat(L, rest...)
    (; peak = maximum(L))
end
# post hoc: no argument list to read, so 1-arg by default and an EXPLICIT coords opt-in.
_tt_posthoc_plain(y) = (; peak = maximum(y))
@kernel (:time => :stat) _tt_posthoc_plain
_tt_posthoc_coords(y, t) = (; tmax = t[argmax(y)])
@kernel (:time => :stat) coords=true _tt_posthoc_coords

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

    # The product-`map` override is N-generic, but was only ever exercised at arity 2
    # (treearrays-use §7, the QT twin) and at arity 4 by `_tt_stats_percentiles` -- whose
    # `f` IGNORES its coordinates, so only the SHAPE was covered, never the value mapping.
    # Bruno sweeps 6 / 8 / 10 (`db_profile_plot`). Decision 1qex8lj: pin it.
    @testset "map over Iterators.product of TreeDims is arity-generic (1qex8lj)" begin
        # heterogeneous label types, mirroring Bruno's db_profile_plot
        alldims = [
            TreeDim(:fit, (:a, :b)),           TreeDim(:health, ("hi", "lo")),
            TreeDim(:source, 1:2),             TreeDim(:outcome, (:x, :y)),
            TreeDim(:draw_selection, (:all,)), TreeDim(:vessel, ("v1", "v2")),
            TreeDim(:diet, (:std,)),           TreeDim(:schedule, ("s",)),
            TreeDim(:dose, (20, 200)),         TreeDim(:subject, 1:3),
        ]
        for n in (1, 3, 6, 8, 10)
            ds = alldims[1:n]
            r = map(Iterators.product(ds...)) do lvals
                join(string.(lvals), "|")           # f receives the coordinate tuple, in order
            end
            @test r isa TreeData
            @test size(parent(r)) == Tuple(length.(ds))
            @test map(TreeArrays.name, TreeArrays.dims(r)) == Tuple(TreeArrays.name.(ds))
            # cell [i,j,…] is the coords at those positions, in dim order -- not just the shape
            @test parent(r) == [join(string.(c), "|") for c in
                Iterators.product((TreeArrays.meta(d).values for d in ds)...)]
        end

        # Bruno's real shape: every cell is a reduced TreeData leaf, nothing densified
        sweep = map(Iterators.product(alldims...)) do lvals
            quantile(TreeData(randn(8, 4), :draw, :time), TreeDim(:q, (0.1, 0.5, 0.9)); dims=:draw)
        end
        @test size(parent(sweep)) == Tuple(length.(alldims))
        @test parent(sweep)[1] isa TreeData

        # the `_reduceouter` positional-axis assertion must not fire on a product-map result
        plain = map(Iterators.product(alldims[1:3]...)) do lvals; float(length(string(lvals))); end
        @test mapslices(sum, plain; dims=:fit) isa TreeData

        # a SCALAR dim is a fixed position, not an axis: it contributes no array axis and
        # lands as a trailing fixed dim (`oni1bc`). Wrap in a 1-tuple to sweep it.
        sc = map(Iterators.product(TreeDim(:fit, (:a, :b)), TreeDim(:dose, 20))) do lvals; string(lvals); end
        @test size(parent(sc)) == (2,)                              # :dose adds no axis
        @test !TreeArrays._isaxis(TreeArrays.dims(sc)[2])           # ... it is fixed
        @test TreeArrays._isaxis(TreeArrays.dims(sc)[1])

        # an UNLABELLED dim has no coordinates to sweep -- say so, don't surface Base's
        # `MethodError: no method matching length(::Missing)` from inside Iterators.product
        @test_throws "no coordinates to sweep" map(
            Iterators.product(TreeDim(:fit, (:a, :b)), TreeDim(:dose))) do lvals; lvals; end
        @test_throws "no coordinates to sweep" map(identity, TreeDim(:dose))
    end

    @testset "reduce a TreeTuple container: map(f, ::TreeDim) reduces like its product twin (snag nanquantile-tree)" begin
        # `map(f, ::TreeDim)` (setdim.jl) builds a TreeTuple whose ELEMENTS are TreeData -- the
        # single-axis twin of the product-swept TreeRaggedArray above. It must reduce identically.
        # Reporter's exact shape: an outer :covariate TreeTuple of aligned dense TreeArray leaves.
        mk() = map(Iterators.product(TreeDim(:parameter, ("Vc",)), TreeDim(:draw, Base.OneTo(2)),
                                     TreeDim(:cov_value, 1:2))) do (_, draw, value)
            Float64(draw + value)
        end
        xt = map(TreeDim(:covariate, ("Age", "Weight"))) do _; mk(); end                   # Tuple parent -> TreeTuple
        xr = map(Iterators.product(TreeDim(:covariate, ("Age", "Weight")))) do (_,); mk(); end  # Array parent -> ragged

        @test parent(xt) isa Tuple && xt isa TreeArrays.TreeTuple
        @test TreeArrays._eltype(xt) == Float64                                             # was: the TreeArray container type

        rows(td) = sort(collect(Tables.rowtable(td)), by = string)
        # reporter's literal call: nanquantile with NamedTuple probs over :draw. No throw; native.
        rt = nanquantile(xt, (median = 0.5,); dims = :draw)
        @test rt isa TreeData
        @test Set(map(TreeArrays.name, TreeArrays.dims(rt))) ⊇ Set((:covariate,))           # outer axis preserved
        @test Tables.columnnames(Tables.columns(rt)) == (:covariate, :parameter, :cov_value, :median)

        # inner reduce keeps the Tuple parent (preserving the heterogeneity-typed container)...
        @test parent(mean(xt; dims = :draw)) isa Tuple
        # ...and is byte-for-byte the product-swept twin, for every reducer that hit the bug.
        for (ft, fr) in (
            (() -> nanquantile(xt, (median=0.5, q025=0.025, q975=0.975); dims=:draw),
             () -> nanquantile(xr, (median=0.5, q025=0.025, q975=0.975); dims=:draw)),
            (() -> nanquantile(xt, TreeDim(:pct, (0.25, 0.5, 0.75)); dims=:draw),
             () -> nanquantile(xr, TreeDim(:pct, (0.25, 0.5, 0.75)); dims=:draw)),
            (() -> quantile(xt, (median=0.5,); dims=:draw),
             () -> quantile(xr, (median=0.5,); dims=:draw)),
            (() -> mean(xt; dims=:draw), () -> mean(xr; dims=:draw)),
            (() -> sum(xt; dims=:draw),  () -> sum(xr; dims=:draw)),
            (() -> std(xt; dims=:draw),  () -> std(xr; dims=:draw)),
            (() -> mapslices(maximum, xt; dims=:draw), () -> mapslices(maximum, xr; dims=:draw)),
        )
            @test rows(ft()) == rows(fr())
        end

        # reducing the OUTER positional axis delegates to the ragged machinery -> same answer.
        @test rows(mean(xt; dims = :covariate)) == rows(mean(xr; dims = :covariate))

        # a scalar-tuple LEAF (a quantile band result) still reduces its own positional axis.
        band = quantile(TreeData(randn(50), :draw), TreeDim(:band, (0.1, 0.5, 0.9)); dims = :draw)
        @test parent(band) isa Tuple && mean(band; dims = :band) isa TreeData
    end

    @testset "setdim stubs throw instead of silently no-oping" begin
        X = TreeData(randn(4,3), :draw, :param)
        @test (try; setdim(X; foo=:bar); false; catch; true; end)                # was: silently returned X unchanged
        @test (try; setdim((:draw, :param); foo=:bar); false; catch; true; end)  # was: bare error() with no message
    end

    @testset "selectdim — restrict a named axis by a label predicate (snag named-axis-label)" begin
        # Reporter's exact shape: TreeData(mat, :draw, :chain, :param => names); :draw/:chain unlabelled.
        axvals(x, i) = TreeArrays.meta(TreeArrays.dims(x)[i]).values
        names = vcat(["unit_params_covariate_effects.$i" for i in 1:5], ["sigma", "lp__", "beta"])
        mat   = reshape(collect(1.0:(6*2*8)), 6, 2, 8)
        X     = TreeData(mat, :draw, :chain, :param => names)
        pat   = r"^unit_params_covariate_effects\."
        mask  = map(contains(pat), names)                       # what draws_core's subdf_pattern computes

        # Regex — the subdf(pattern::Regex) case: sub-tree over the matched labels, others intact.
        Y = selectdim(X, :param => pat)
        @test parent(Y) isa SubArray && parent(Y) == mat[:, :, mask]        # NO data copy: a view
        @test parent(Y).parent === mat                                     # shares the original's memory
        @test collect(axvals(Y, 3)) == names[mask]
        @test size(parent(Y)) == (6, 2, 5)
        @test map(TreeArrays.name, TreeArrays.dims(Y)) == (:draw, :chain, :param)   # other axes intact

        @test parent(selectdim(X, :param, pat)) == parent(Y)               # 3-arg form == Pair form
        @test all(startswith("sigma"), axvals(selectdim(X, :param => startswith("sigma")), 3))
        @test parent(selectdim(X, :param => mask)) == mat[:, :, mask]       # explicit Bool mask
        @test parent(selectdim(X, :param => [8, 1])) == mat[:, :, [8, 1]]   # explicit integer index (reorder)
        @test size(parent(selectdim(X, :chain => [1]))) == (6, 1, 8)       # positional select of an UNLABELLED axis
        @test size(parent(selectdim(X, :param => r"^nomatch"))) == (6, 2, 0)   # empty match = zero-width axis, no error

        # loud, never silently wrong:
        @test_throws "no axis named" selectdim(X, :prm => pat)                    # typo
        @test_throws "ambiguous" selectdim(X, :param => "beta")                   # bare String
        @test_throws "must return Bool" selectdim(X, :param => (s -> length(s)))  # predicate returns Int
        @test_throws "Bool mask has length" selectdim(X, :param => [true, false]) # wrong-length mask
        @test_throws "unlabelled axis" selectdim(X, :draw => pat)                 # Regex on an unlabelled axis

        # Tuple axis stays a Tuple; select ∘ reduce chains and stays a TreeData.
        T = TreeData(randn(3, 4), :a => (:x, :y, :z), :b)
        @test axvals(selectdim(T, :a => in((:x, :z))), 1) === (:x, :z)
        @test size(parent(mean(selectdim(X, :param => pat); dims = :draw))) == (2, 5)
    end

    @testset "outer_dim survives TreeData-forwarding" begin
        Xnt = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(4), :draw)))
        Y = TreeData(Xnt, TreeDim(:extra, nothing))
        @test outerdim(Y) === outerdim(Xnt)                    # was: no `outer_dim` field in meta(Y)
        Z = mapslices(mean, Y; dims=:draw)                      # was: errors calling outerdim(Y) inside mapslices
    end

    # `post.beta` reads a record field. Same descent as `_mapslices(::TreeNamedTuple)`:
    # the record axis is split off, the field gets the container's inner axes, nothing
    # densifies. Inference is checked through a function barrier (how it's really used).
    @testset "getproperty reads record fields, zero-copy + type-stable" begin
        geta(X) = X.a
        getbeta(X) = X.beta

        Xnt = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(4), :draw)))
        @test Xnt.a === parent(Xnt).a                     # already a TreeData -> returned as is
        @test propertynames(Xnt) == (:a, :b)
        @test (@inferred geta(Xnt)) === parent(Xnt).a

        # a RAW field is wrapped with the inner axes: :draw survives, record axis drops
        raw = (;beta=randn(4), sigma=randn(4))
        Xr = TreeData(:param => raw, TreeDim(:draw, 1:4))
        @test parent(Xr.beta) === raw.beta                # zero-copy: the same backing array
        @test map(TreeArrays.name, TreeArrays.dims(Xr.beta)) == (:draw,)
        @test (@inferred getbeta(Xr)) isa TreeData

        # a ghost dim stays on the container, never on the child
        @test TreeData(Xnt, TreeDim(:extra, nothing)).a === parent(Xnt).a

        @test_throws ArgumentError Xnt.nope               # never a silent `missing`/`nothing`
        @test parent(Xnt) isa NamedTuple                  # internals still reach fields via getfield
        @test TreeArrays.meta(Xnt) isa NamedTuple
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

        # Reducing the sole remaining axis (:param) used to WRAP the leaf in an extra
        # TreeData{<:TreeData} level (a use-less artifact). `_assemble` now merges the reduced-as-
        # ghost dims into the leaf instead, so the Tuple/scalar sits one level shallower --
        # parent(chained), not parent(parent(·)). Nothing is lost: the ghosts still ride the leaf
        # (dims == (:tup, :param, :draw)); it just isn't nested behind a wrapper anymore.
        chained = quantile(tuple_result, TreeDim(:tup2, (0.1, 0.9)); dims=:param)  # was: MethodError CartesianIndices(::Tuple)
        @test parent(chained) isa Tuple

        chained_scalar = quantile(scalar_result, TreeDim(:median2, 0.5); dims=:param)  # was: MethodError CartesianIndices(::Number)
        @test parent(chained_scalar) isa Number
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

    # quantile(X, :band => spec; dims) -- decision e18kfn / option A': a name<->prob spec reduces
    # to a band AXIS labeled by the NAMES (keys, Symbols) and computed from the PROBS (values).
    # Distinct from the p::NamedTuple record above -- orientation stays a VIEW choice (long here;
    # TreeTable(wide=:band) will pivot it), nothing baked wide at reduction time. No dim-model
    # change: `meta.values` already carries the levels, so an ordinary Symbol-Tuple axis suffices.
    @testset "quantile(X, :band => spec; dims) -> Symbol-labeled band axis (A')" begin
        Xq = TreeData(reshape(1.0:12.0, 4, 3), :draw, :param)
        spec = (median=0.5, q025=0.025, q975=0.975)

        r = quantile(Xq, :band => spec; dims=:draw)
        leaf = first(parent(r))
        bandaxis = TreeArrays.dims(leaf)[1]
        @test TreeArrays.name(bandaxis) === :band
        @test TreeArrays.meta(bandaxis).values == keys(spec)              # axis labels = the NAMES
        @test TreeArrays._isaxis(bandaxis)                               # an ordinary axis
        @test parent(leaf) == Statistics.quantile(1.0:4.0, values(spec))  # computed from the PROBS
        for j in 1:3
            @test parent(parent(r)[j]) == Statistics.quantile(Float64.(4j-3:4j), values(spec))
        end

        # the existing LONG melt handles it for free: one :band column of the level names.
        cols = Tables.columns(r)
        @test Set(Tables.columnnames(cols)) == Set((:param, :band, :value))
        @test Set(Tables.getcolumn(cols, :band)) == Set(keys(spec))
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

        @testset ":band => spec -> Symbol-labeled band axis (A'; mirrors quantile Pair method)" begin
            spec = (median=0.5, q025=0.25, q975=0.75)
            r = nanquantile(Xq, :band => spec; dims=:draw)
            leaf = first(parent(r))
            @test TreeArrays.meta(TreeArrays.dims(leaf)[1]).values == keys(spec)  # axis labels = NAMES
            @test parent(leaf) == parent(first(parent(quantile(Xq, :band => spec; dims=:draw))))  # no-NaN match

            P = Array(reshape(1.0:12.0, 4, 3)); P[:, 1] .= NaN
            Xn2 = TreeData(P, :draw, :param)
            leafn = first(parent(nanquantile(Xn2, :band => spec; dims=:draw)))
            @test all(isnan, parent(leafn))   # all-NaN slice -> NaN at every level, never throws
        end
    end

    # A reduction is NOT a precondition for table-ness, and a coordinate axis is not
    # restricted to numeric levels -- `treearrays-use` §9 implied both. The spaghetti
    # consumer (per-draw lines, `group=:draw`) needs an UNREDUCED tree whose `:draw`
    # axis carries String ids that stay unique across product-sweep cells.
    @testset "Tables.jl: an UNREDUCED tree melts long; String axis levels survive" begin
        ndraw, ntime = 4, 3
        vals     = reshape(collect(1.0:ndraw*ntime), ndraw, ntime)
        draw_ids = ["comboA_$d" for d in 1:ndraw]
        times    = [0.0, 0.5, 1.0]
        td = TreeData(vals, :draw => draw_ids, :time_h => times)   # no reduction applied

        @test Tables.istable(typeof(td))
        sch = Tables.schema(td)
        @test sch.names == (:draw, :time_h, :value)
        @test sch.types == (String, Float64, Float64)

        cols = Tables.columns(td)
        @test Tables.columnnames(cols) == sch.names
        # the String axis stays a lazy, concretely-typed view -- never materialized
        @test Tables.getcolumn(cols, :draw) isa TreeArrays.AxisColumn{String}
        @test Tables.getcolumn(cols, :value) isa TreeArrays.ValueColumn{Float64}
        @test length(Tables.getcolumn(cols, :value)) == ndraw * ntime
        # one row per (draw, time); every draw id recurs once per time point, so a
        # `group=:draw` channel yields exactly `ndraw` lines and fuses none of them.
        @test length(unique(Tables.getcolumn(cols, :draw))) == ndraw
        @test collect(Tables.getcolumn(cols, :draw))[1:ndraw] == draw_ids
        @test collect(Tables.getcolumn(cols, :value)) == vec(vals)

        # wide-vs-long is decided by the BACKING CONTAINER, not by the label type:
        # Symbol levels on an array-backed axis still melt LONG (a `:band` column)...
        symaxis = TreeData([1.0, 2.0, 3.0], :band => (:lower, :median, :upper))
        @test Tables.columnnames(Tables.columns(symaxis)) == (:band, :value)
        @test Tables.getcolumn(Tables.columns(symaxis), :band) isa TreeArrays.AxisColumn{Symbol}
        # ...while a NamedTuple *record* parent, same labels, goes WIDE.
        rec = TreeData(:quantile => (lower=1.0, median=2.0, upper=3.0))
        @test Tables.columnnames(Tables.columns(rec)) == (:lower, :median, :upper)
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

    # A ragged tree melts LONG: the row count is `sum(length, leaves)`, not a product
    # of axis lengths, and each row's coordinate is read from the position it occupies
    # in its OWN subtree. Long form never needed rectangularity -- only the `wide=`
    # pivot does (a pivot needs ONE level set and ONE column length), and that stays
    # refused below. This is the shape `treearrays-use` §8 names as the ragged case.
    @testset "Tables.jl: ragged trees melt LONG (rows are a SUM, not a product)" begin
        # matches docs/pkpd_demo.jl's input_data shape: per-subject arrays of
        # genuinely differing length under an outer :subject axis.
        n_subjects = 4
        n_measurements = [3, 5, 3, 3]   # subject 2 differs -- ragged
        times = [sort(randn(n)) for n in n_measurements]
        vals  = [randn(n) for n in n_measurements]
        ragged = TreeData(map(TreeData(:time), vals, times), :subject)

        @test Tables.schema(ragged) isa Tables.Schema   # type-only -- physically cannot see instance raggedness
        @test Tables.schema(ragged).names == (:subject, :time, :value)
        cols = Tables.columns(ragged)

        # THE point: 14 rows, not a 4 x ? product. Nothing was densified to get here.
        @test length(Tables.getcolumn(cols, :value)) == sum(n_measurements) == 14
        # each sibling's rows are contiguous, so every column is its per-leaf
        # concatenation -- the melt a consumer would otherwise hand-roll with
        # `reduce(vcat, ...)` + `fill(label, n)`.
        @test Tables.getcolumn(cols, :value) == reduce(vcat, vals)
        @test Tables.getcolumn(cols, :time)  == reduce(vcat, times)
        # `:subject` is an UNLABELLED axis here, so its coordinate is the index --
        # repeated exactly its own leaf's length, never smeared to a common width.
        @test Tables.getcolumn(cols, :subject) ==
              reduce(vcat, [fill(s, n) for (s, n) in enumerate(n_measurements)])

        # a ragged boundary cannot read `:time` off one representative, so that column
        # walks per row; the OUTER axis still reads its own plan slot directly.
        @test Tables.getcolumn(cols, :time) isa TreeArrays.WalkAxisColumn
        @test Tables.getcolumn(cols, :subject) isa TreeArrays.AxisColumn

        # the regular (same length, identical coordinates) counterpart is UNCHANGED:
        # same dense product row space, same O(1)-storage shared columns.
        shared_times = sort(randn(3))
        regular = TreeData(map(_ -> TreeData(randn(3), :time => shared_times), 1:n_subjects), :subject)
        rcols = Tables.columns(regular)
        @test length(Tables.getcolumn(rcols, :value)) == n_subjects * 3
        @test Tables.getcolumn(rcols, :time) isa TreeArrays.AxisColumn
    end

    # The `===`-then-`isequal` coordinate signature no longer decides ACCEPT vs ERROR --
    # it decides SHARED vs PER-ROW columns. Siblings that agree keep the O(1)-storage
    # `AxisColumn`/`ConstColumn` this adapter has always built; siblings that diverge get
    # `Walk*` columns that re-read the dim from the sibling each row lands in. So the two
    # false-positive cases still matter exactly as much: a miss is now a silent storage
    # and speed downgrade on a perfectly regular tree instead of a spurious error.
    @testset "Tables.jl: coordinate-value guard (===-fast-path + isequal fallback, Option A)" begin
        # false positive 1: siblings share the SAME coordinate object -- every
        # comparison is an `===` hit, O(1), zero `isequal` calls needed.
        shared_times = sort(randn(3))
        shared_obj = TreeData(map(_ -> TreeData(randn(3), :time => shared_times), 1:4), :subject)
        @test length(Tables.getcolumn(Tables.columns(shared_obj), :value)) == 12
        @test Tables.getcolumn(Tables.columns(shared_obj), :time) isa TreeArrays.AxisColumn

        # false positive 2: siblings each get their OWN, independently-built but
        # value-equal coordinate array -- `===` misses, `isequal` accepts, so this
        # is still recognised as regular and still gets ONE shared column.
        distinct_but_equal = TreeData(map(_ -> TreeData(randn(3), :time => copy(shared_times)), 1:4), :subject)
        @test length(Tables.getcolumn(Tables.columns(distinct_but_equal), :value)) == 12
        @test Tables.getcolumn(Tables.columns(distinct_but_equal), :time) isa TreeArrays.AxisColumn

        # divergence, single-level: same lengths, genuinely different per-sibling
        # coordinate VALUES. The row space IS still a product (so the dense decode and
        # the pre-existing row order are kept -- outermost varies FASTEST), but `:time`
        # cannot be read off one representative, so it walks.
        dtimes = [sort(randn(3)) for _ in 1:4]
        dvals  = [randn(3) for _ in 1:4]
        differing = TreeData(map(TreeData(:time), dvals, dtimes), :subject)
        @test Tables.schema(differing) isa Tables.Schema   # schema still succeeds -- type-only
        dcols = Tables.columns(differing)
        @test length(Tables.getcolumn(dcols, :value)) == 12
        @test Tables.getcolumn(dcols, :time) isa TreeArrays.WalkAxisColumn
        @test Tables.getcolumn(dcols, :time)  == vec([dtimes[s][k] for s in 1:4, k in 1:3])
        @test Tables.getcolumn(dcols, :value) == vec([dvals[s][k]  for s in 1:4, k in 1:3])

        # divergence, multi-level: TWO top-level siblings, each internally uniform (all of
        # subject A's visits share ONE :time array, all of subject B's visits share a
        # DIFFERENT one) -- an own-level-only check would miss this (each boundary's own
        # siblings look consistent); the full recursive coordinate signature catches it via
        # the representative (first-visit) path comparison at the TOP boundary. Checked
        # per row rather than per column: a coordinate smeared from the representative
        # would still produce 12 plausible rows, just ones whose `:time` and `:value` come
        # from different leaves. Pinning them TOGETHER is what catches that.
        time_a, time_b = sort(randn(2)), sort(randn(2))
        subject_a = TreeData([TreeData(randn(2), :time => time_a) for _ in 1:3], :visit)
        subject_b = TreeData([TreeData(randn(2), :time => time_b) for _ in 1:3], :visit)
        nested_differing = TreeData([subject_a, subject_b], :subject)
        nrt = Tables.rowtable(nested_differing)
        @test length(nrt) == 2 * 3 * 2
        @test all(nrt) do r
            tv = r.subject == 1 ? time_a : time_b
            k  = findfirst(==(r.time), tv)
            leaf = parent(parent(nested_differing)[r.subject])[r.visit]
            k !== nothing && r.value == parent(leaf)[k]
        end

        # the multi-level REGULAR counterpart (both subjects share ONE :time
        # object across all visits, all subjects) still gets shared columns.
        time_shared = sort(randn(2))
        subject_a2 = TreeData([TreeData(randn(2), :time => time_shared) for _ in 1:3], :visit)
        subject_b2 = TreeData([TreeData(randn(2), :time => time_shared) for _ in 1:3], :visit)
        nested_regular = TreeData([subject_a2, subject_b2], :subject)
        @test length(Tables.getcolumn(Tables.columns(nested_regular), :value)) == 2 * 3 * 2
        @test Tables.getcolumn(Tables.columns(nested_regular), :time) isa TreeArrays.AxisColumn
    end

    # The correctness case the snag's ask does not itself name, but which the same
    # boundary decides: once a boundary is KNOWN divergent, the FIXED dims below it must
    # be read per-sibling too. Reading `dose` off the representative would print subject
    # 1's dose on every subject's rows -- silently reintroducing the metadata smear (§6)
    # that TreeArrays exists to retire, at exactly the boundary TreeArrays owns.
    #
    # Scope, precisely: divergence is detected from SHAPE or from AXIS coordinates, and
    # the fixed dims then follow that verdict. A cohort agreeing on BOTH and differing
    # only on a fixed value is still smeared -- decision 4b3vcd's beta case, unchanged
    # and deliberate (detecting it needs a per-element walk, which is O(rows) -- see
    # `_coordsig`). Pinned at the bottom so that boundary stays visible in the tests
    # rather than reading like an oversight.
    @testset "Tables.jl: per-sibling FIXED dims are read per row, never smeared (ragged)" begin
        doses = [20, 200, 20]
        vals  = [randn(n) for n in (2, 4, 3)]
        times = [sort(randn(n)) for n in (2, 4, 3)]
        cohort = TreeData([TreeData(vals[s], :time => times[s]; dose = doses[s]) for s in 1:3],
                          :subject => ["s1", "s2", "s3"])

        cols = Tables.columns(cohort)
        @test Tables.schema(cohort).names == (:subject, :time, :dose, :value)
        @test length(Tables.getcolumn(cols, :value)) == 9
        # each subject's own dose, repeated exactly its own leaf's length
        @test Tables.getcolumn(cols, :dose) == reduce(vcat, [fill(d, length(v)) for (d, v) in zip(doses, vals)])
        @test Tables.getcolumn(cols, :dose) isa TreeArrays.WalkConstColumn
        @test Tables.getcolumn(cols, :subject) == reduce(vcat, [fill("s$s", length(vals[s])) for s in 1:3])
        @test Tables.getcolumn(cols, :value) == reduce(vcat, vals)
        @test Tables.getcolumn(cols, :time)  == reduce(vcat, times)

        # a cohort that AGREES on the fixed dim keeps the de-duplicated O(1) ConstColumn
        agree = TreeData([TreeData(randn(3), :time => 1:3; dose = 20) for _ in 1:3],
                         :subject => ["s1", "s2", "s3"])
        @test Tables.getcolumn(Tables.columns(agree), :dose) isa TreeArrays.ConstColumn

        # 4b3vcd beta, UNCHANGED: identical shape AND identical axis coords, differing
        # fixed value only -> the divergence is not detected, and the representative's
        # value is reported for every row. This is the documented boundary, not a
        # regression of the above: express a per-sibling value as an AXIS coordinate.
        smeared = TreeData([TreeData(randn(3), :time => 1:3; dose = 10d) for d in 1:3],
                           :subject => ["s1", "s2", "s3"])
        @test Tables.getcolumn(Tables.columns(smeared), :dose) == fill(10, 9)
    end

    # The lazy-column invariant (decision 1uzarfr) has to survive the ragged path too:
    # a `Walk*` column is still a view-with-a-rule, so `Tables.columns` must stay
    # O(structure) -- the ragged offset table is O(siblings), never O(rows).
    @testset "Tables.jl: ragged melt keeps columns LAZY (rows scale, allocation does not)" begin
        mkragged(n_rows) = TreeData([TreeData(randn(n_rows + s), :time => sort(randn(n_rows + s)))
                                     for s in 1:4], :subject)
        small, big = mkragged(10), mkragged(1000)
        Tables.columns(small); Tables.columns(big)          # warm up / compile
        a_small = @allocated Tables.columns(small)
        a_big   = @allocated Tables.columns(big)
        # 100x the rows, same structure: allocation must stay FLAT (not scale with rows)
        @test a_big <= a_small * 4
        @test length(Tables.getcolumn(Tables.columns(big), :value)) == sum(1001:1004)
        # and it really is a view -- no row data was materialized to build it
        @test Tables.getcolumn(Tables.columns(big), :value) isa TreeArrays.ValueColumn
        @info "ragged melt laziness gate: Tables.columns(X) @allocated (10 vs 1000 rows/leaf)" a_small a_big
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

    # `showable(MIME"text/html"(), x)` is the branch HTMX's builder loop takes before
    # falling back to `print(io, child)`, so these methods are the whole of "automatic
    # rich display". Every preview is bounded -- a display method must never densify.
    @testset "rich HTML display is automatic, escaped, and bounded" begin
        html(x) = sprint(show, MIME"text/html"(), x)

        X = TreeData(randn(4, 3), :draw, :param)
        @test showable(MIME"text/html"(), X)          # the chokepoint
        @test occursin("<code>draw</code>", html(X)) && occursin("<code>param</code>", html(X))
        @test occursin("TreeArray", html(X))

        # HTMX splices a showable child's output RAW (HTMX.jl:202) -- the emitter owns
        # escaping, so coords carrying HTML metacharacters must not break the DOM.
        Xesc = TreeData(randn(2), TreeDim(:tag, ("<b>&x", "y")))
        @test occursin("&lt;b&gt;&amp;x", html(Xesc))
        @test !occursin("<b>", html(Xesc))

        # `_esc` is context-free: the full `& < > " '` set, so a future edit that puts
        # an escaped value in an ATTRIBUTE is safe by construction, not by memory.
        @test TreeArrays._esc("""<a href="x" class='y'>&""") ==
              "&lt;a href=&quot;x&quot; class=&#39;y&#39;&gt;&amp;"
        @test TreeArrays._esc("&amp;") == "&amp;amp;"      # single-pass: no double-escape

        # a record: one collapsible section per field, outer_dim labelled `record`
        Xnt = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(4), :draw)))
        @test count("<details>", html(Xnt)) == 2
        @test occursin("record", html(Xnt))

        # ragged: bounded leaf preview that SAYS how many it skipped (never silent)
        hr = html(TreeData([TreeData(randn(2, 3), :time, :chan) for _ in 1:5], :subject))
        @test count("<details>", hr) == TreeArrays._MAX_LEAVES
        @test occursin("2 more leaves", hr)

        # TreeTable renders real rows off the lazy melt, capped and self-announcing
        big = TreeData(reshape(1.0:150.0, 50, 3), :draw, :param)
        ht = html(TreeTable(big))
        @test occursin("150 rows", ht) && occursin("showing the first 20", ht)
        @test count("<tr>", ht) == TreeArrays._MAX_ROWS + 1        # header + body

        hs = html(TreeTable(TreeData(reshape(1.0:6.0, 2, 3), :draw, :param)))
        @test occursin("6 rows", hs) && !occursin("showing the first", hs)
        @test count("<tr>", hs) == 7

        # a wide pivot renders too: its levels are the columns, rows drop to the draws
        hw = html(TreeTable(big; wide=:param))
        @test occursin("param_1", hw) && occursin("50 rows", hw)

        # but a wide dim that has no levels to spread still throws out of `show` --
        # display is not where a bad shape gets to render as if it were fine
        fixedX = TreeData(randn(4), TreeDim(:draw, 1:4), TreeDim(:tag, :fixedtag))
        @test_throws "not a real axis" html(TreeTable(fixedX; wide=:tag))

        # a RAGGED tree IS a table in long form, so its display renders -- 2+5+9 rows,
        # each with its own `:time`. What still throws is the ragged tree under `wide=`
        # (below): display is not where a bad shape gets to render as if it were fine,
        # but a good shape must not be refused there either.
        ragged = TreeData([TreeData(randn(n), :time => sort(randn(n))) for n in (2, 5, 9)], :subject)
        rh = html(TreeTable(ragged))
        @test occursin("16 rows", rh) && occursin("time", rh)
        @test_throws "not a supported Tables shape under `wide=`" html(TreeTable(ragged; wide=:time))
        @test occursin("more leaves", html(ragged)) == false     # 3 leaves, none skipped
        @test count("<details>", html(ragged)) == 3
    end

    # HTMX's markdown rep has a `show(io, ::MIME"text/markdown", val) = print(io, string(val))`
    # CATCH-ALL and no `showable` guard, so `?plain` rendered a TreeData as `string(td)`.
    # Two bugs, not one: no markdown method here, AND `print_tree` set no `:limit`, so
    # `string`/`@info` dumped the whole backing array. The REPL sets `:limit` itself,
    # which is why display always looked fine.
    @testset "markdown display + print_tree is bounded at the root" begin
        md(x) = sprint(show, MIME"text/markdown"(), x)
        big = TreeData(randn(1000, 100), :draw, :param)

        @test length(string(big)) < 2000          # was 916_987 chars
        @test occursin("…", string(big))          # Base marks its own elision -- never silent
        # an explicit :limit=>false still gets the full dump: the escape hatch survives
        @test length(sprint(io -> print(IOContext(io, :limit => false), big))) > 100_000

        # the array-backed leaf was only ONE of three unbounded holes on this path.
        # `:limit` does not apply to Tuples, and `show(::TreeDim)` set no `:limit` at all,
        # so a big coord list or a Tuple leaf still dumped tens of KB through `print_dims`.
        bigtup = Tuple(Symbol("subject_", i) for i in 1:4000)
        @test length(string(TreeData(randn(4000), TreeDim(:subject, bigtup)))) < 2000    # was 59_164
        @test occursin("…", string(TreeData(randn(4000), TreeDim(:subject, bigtup))))    # elision marked
        bigvec = [Symbol("s", i) for i in 1:4000]
        @test length(string(TreeData(randn(4000), TreeDim(:subject, bigvec)))) < 2000    # was 31_161
        tup = TreeData(Tuple(1.0*i for i in 1:3000), TreeDim(:k, 1:3000))
        @test length(string(tup)) < 2000                                                 # was 22_950
        @test occursin("…", string(tup))
        # the compact path (`print_tree`'s `compact && return print_values(cio, X)`) too
        @test length(sprint(io -> show(IOContext(io, :compact => true), big))) < 2000

        @test showable(MIME"text/markdown"(), big)
        @test showable(MIME"text/markdown"(), TreeTable(big))
        m = md(big)
        @test length(m) < 2000
        @test occursin("**TreeArray**", m) && occursin("`draw`", m) && occursin("`param`", m)

        # a `|` in a coordinate closes a cell -- it must be escaped
        @test occursin("a\\|b", md(TreeData(randn(2), TreeDim(:tag, ("a|b", "c")))))

        # ... and so does a newline, which closes the whole ROW. Cell values come from
        # `print` (raw), unlike the show-derived coord previews, so they need their own
        # escaper -- `| a⏎b |` split the table into two broken rows.
        nl = md(TreeTable(TreeData(randn(2), TreeDim(:tag, ("a\nb", "c")))))
        @test !occursin("a\nb", nl) && occursin("a\\nb", nl)
        @test count("\n|", nl) == 2 + 2                     # header + separator + 2 rows, not 3
        # a value ending in `\` would otherwise escape the cell delimiter emitted after it
        @test occursin("ends\\\\ |", md(TreeTable(TreeData(randn(1), TreeDim(:tag, ("ends\\",))))))
        # show-derived text must NOT be backslash-escaped again (`a\nb` -> `a\\nb`)
        @test occursin("\"a\\nb\"", md(TreeData(randn(2), TreeDim(:tag, ["a\nb", "c"]))))

        Xnt = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(4), :draw)))
        @test occursin("**a**", md(Xnt)) && occursin("**b**", md(Xnt))
        @test occursin("2 more leaves", md(TreeData([TreeData(randn(2,3), :time, :chan) for _ in 1:5], :subject)))

        spec = (lower=0.25, median=0.5, upper=0.75)
        r = quantile(TreeData(reshape(1.0:12.0, 4, 3), :draw, :param), :band => spec; dims=:draw)
        t = md(TreeTable(r; wide=:band))
        @test occursin("**TreeTable** — 3 rows", t)
        @test occursin("`lower`", t) && occursin("`upper`", t)
        @test count("\n|", t) == 2 + 3            # header + separator + 3 body rows

        tbig = md(TreeTable(TreeData(reshape(1.0:150.0, 50, 3), :draw, :param)))
        @test occursin("150 rows", tbig) && occursin("showing the first 20", tbig)
    end

    # `wide=:band` spreads an axis's levels into columns. It is a pure RE-INDEXING of
    # the columns the long melt already built -- same lazy column objects, one slot
    # deleted from the shared row space -- so nothing recomputes and nothing densifies.
    # This is the shape AoV's `lineribbon(bands=[:lower => :upper])` consumes.
    @testset "TreeTable(wide=) pivots an axis's levels into columns" begin
        spec = (lower=0.25, median=0.5, upper=0.75)
        Xq = TreeData(reshape(1.0:12.0, 4, 3), :draw, :param)
        r  = quantile(Xq, :band => spec; dims=:draw)

        @test Set(Tables.columnnames(Tables.columns(TreeTable(r)))) == Set((:param, :band, :value))

        # `:band` sits on the LEAF, not at the top level -- validate + pivot anyway
        tt = TreeTable(r; wide=:band)
        cols = Tables.columns(tt)
        @test Set(Tables.columnnames(cols)) == Set((:param, :lower, :median, :upper))
        # wide mode returns a STORED (runtime) schema -- names are coordinate values,
        # not type parameters, so `.names` is a Vector here and a Tuple in long mode.
        # Both are valid `Tables.Schema`s (decision 1krjg6l).
        sch = Tables.schema(tt)
        @test Tuple(sch.names) == Tables.columnnames(cols)
        @test sch.names isa Vector{Symbol}
        @test Tables.schema(TreeTable(r)).names isa Tuple      # long mode: type-level, unchanged
        @test all(isconcretetype ∘ eltype, values(cols))     # aov-use §2 column contract

        # the band axis leaves the row space: 3*3 long rows collapse to 3 wide rows
        @test length(Tables.rowtable(TreeTable(r))) == 9
        @test length(Tables.rowtable(tt)) == 3

        # every wide cell equals the long melt's corresponding cell
        long = Tables.rowtable(TreeTable(r))
        for row in Tables.rowtable(tt), lvl in keys(spec)
            @test getproperty(row, lvl) == only(filter(l -> l.param == row.param && l.band === lvl, long)).value
        end
        # ... and equals a direct quantile of that column
        for (j, row) in enumerate(Tables.rowtable(tt))
            col = Float64.(4j-3:4j)
            @test (row.lower, row.median, row.upper) == Tuple(Statistics.quantile(col, values(spec)))
        end

        @test_throws "is not a dim of this TreeData" TreeTable(r; wide=:nope)

        # an unlabelled axis widens positionally, prefixed by its dim (never a bare `1`)
        wp = Tables.columns(TreeTable(Xq; wide=:param))
        @test Set(Tables.columnnames(wp)) == Set((:draw, :param_1, :param_2, :param_3))
        @test collect(Tables.getcolumn(wp, :param_2)) == collect(5.0:8.0)

        # a fixed/ghost dim has no levels -- loud, never a silent passthrough
        fixedX = TreeData(randn(4), TreeDim(:draw, 1:4), TreeDim(:tag, :fixedtag))
        @test_throws "not a real axis" Tables.columns(TreeTable(fixedX; wide=:tag))

        # Vega-Lite reads a dot in a field name as nested property access (aov-use §9)
        dotted = TreeData(randn(2, 2), :draw, :time => [0.1, 0.25])
        @test Set(Tables.columnnames(Tables.columns(TreeTable(dotted; wide=:time)))) ==
              Set((:draw, :time_0_1, :time_0_25))

        # ... and a SYMBOL level carrying a dot must be sanitized too. This is the
        # reducer's own primary path (`:band => (var"q0.025"=…,)`), and it used to slip
        # through unsanitized -- a `q0.025` column is `datum["q0"]["025"]` to VL, i.e. a
        # silently wrong plot rather than an error.
        symdot = TreeData(randn(2, 2), :draw, :band => (Symbol("q0.025"), Symbol("q0.975")))
        symnames = Tables.columnnames(Tables.columns(TreeTable(symdot; wide=:band)))
        @test Set(symnames) == Set((:draw, :q0_025, :q0_975))
        @test !any(nm -> occursin('.', String(nm)), symnames)

        # a level colliding with an existing column names the pivot AND the culprit,
        # rather than surfacing as Base's bare "duplicate field name in NamedTuple"
        collide = TreeData(randn(2, 2), :lower, :band => (:lower, :upper))
        @test_throws "duplicate column names" Tables.columns(TreeTable(collide; wide=:band))

        # a single row-dim collapses to redrowdims == () -- one row, no axis columns
        solo = TreeData([10.0, 20.0, 30.0], :band => (:lo, :mid, :hi))
        scols = Tables.columns(TreeTable(solo; wide=:band))
        @test Set(Tables.columnnames(scols)) == Set((:lo, :mid, :hi))
        @test length(Tables.getcolumn(scols, :mid)) == 1
        @test only(Tables.getcolumn(scols, :mid)) == 20.0

        # record fields sharing a band axis fan out prefixed, sharing the id column once
        rec = TreeData(:rec => (; a=TreeData(reshape(1.0:6.0, 2, 3), :draw, :band=>(:lo,:mid,:hi)),
                                   b=TreeData(reshape(7.0:12.0, 2, 3), :draw, :band=>(:lo,:mid,:hi))))
        @test Tables.columnnames(Tables.columns(TreeTable(rec; wide=:band))) ==
              (:draw, :a_lo, :a_mid, :a_hi, :b_lo, :b_mid, :b_hi)

        # multi-dim wide is not built; it must say so, not silently widen just one
        # multi-dim wide is built now (1e3figi) -- its own testset below
        @test length(Tables.columnnames(Tables.columns(TreeTable(r; wide=(:band, :param))))) == 9

        # a ragged source stays loud under wide, exactly as it is under long
        rag = TreeData([TreeData(randn(n), :time) for n in (2, 3)], :subject)
        @test_throws "ragged trees are not a supported Tables shape" Tables.columns(TreeTable(rag; wide=:time))

        # THE headline claim: the pivot re-indexes the long melt's lazy columns, so
        # `columns()` is O(structure), never O(rows). A 1000x row increase must not
        # move the allocation. (Mirrors the Delta A gate above; rtol absorbs call noise.)
        wsmall = TreeTable(TreeData(randn(50,    3), :draw, :band=>(:lo,:mid,:hi)); wide=:band)
        wbig   = TreeTable(TreeData(randn(50000, 3), :draw, :band=>(:lo,:mid,:hi)); wide=:band)
        Tables.columns(wsmall); Tables.columns(wbig)          # warm up
        a_wsmall = @allocated Tables.columns(wsmall)
        a_wbig   = @allocated Tables.columns(wbig)
        @info "wide pivot acceptance gate: Tables.columns @allocated (50 vs 50000 draws)" a_wsmall a_wbig
        @test isapprox(a_wbig, a_wsmall; rtol=0.3)
        @test length(Tables.getcolumn(Tables.columns(wbig), :mid)) == 50000
    end

    # The TA-native spelling of a groupby on a NON-axis, per-item key (decision 17y9dpd):
    # make the key STRUCTURE. Group the items into a rectangular grid of ragged cells --
    # each cell a zero-copy `view` into the backing array -- then reduce the pooled axes
    # away. Cells may hold different item counts; after the reduction the tree is regular,
    # so it melts. This is what Bruno's `aggregate_ribbon` groupby becomes.
    @testset "groupby on a per-item key = a grid of ragged cells (17y9dpd)" begin
        P = reshape(1.0:42.0, 6, 7)                      # (draw x subject)
        dose  = [10, 10, 20, 20, 20, 10, 20]             # per-SUBJECT attributes, not axes
        study = [:A, :A, :A, :B, :B, :B, :B]
        spec  = (lower=0.25, median=0.5, upper=0.75)

        # `dims=(:draw, :subject)` pools BOTH axes into one quantile per cell
        one = quantile(TreeData(view(P, :, [1,2,6]), :draw, :subject), :band => spec; dims=(:draw, :subject))
        @test collect(parent(one)) ≈ Statistics.quantile(vec(P[:, [1,2,6]]), collect(values(spec)))

        doses, studies = [10, 20], [:A, :B]
        cells = [TreeData(view(P, :, findall((study .== s) .& (dose .== d))), :draw, :subject)
                 for d in doses, s in studies]
        @test [size(parent(c), 2) for c in cells] == [2 1; 1 3]        # ragged cell widths
        tree = TreeData(cells, TreeDim(:dose_mg, Tuple(doses)), TreeDim(:study, Tuple(studies)))

        tt = TreeTable(quantile(tree, :band => spec; dims=(:draw, :subject)); wide=:band)
        @test Set(Tables.columnnames(Tables.columns(tt))) == Set((:dose_mg, :study, :lower, :median, :upper))
        rows = Tables.rowtable(tt)
        @test length(rows) == 4
        for row in rows      # each cell pools its subjects' draws, exactly as a groupby would
            ref = Statistics.quantile(vec(P[:, findall((study .== row.study) .& (dose .== row.dose_mg))]),
                                      collect(values(spec)))
            @test collect((row.lower, row.median, row.upper)) ≈ ref
        end

        # A truly ragged GRID (study :B lacks a dose :A has) gives sibling TreeDatas whose
        # axis lengths differ AS TYPES, so `[a, b]` widens to a non-concrete eltype and the
        # type-only `_schema` walk hits the ragged tree itself. It used to die with Base's
        # "type NamedTuple has no field dims"; it must name the real problem.
        inner_A = TreeData([TreeData(view(P,:,[1,2]), :draw, :subject), TreeData(view(P,:,[3]), :draw, :subject)],
                           TreeDim(:dose_mg, (10, 20)))
        inner_B = TreeData([TreeData(view(P,:,[4,5,6,7]), :draw, :subject)], TreeDim(:dose_mg, (20,)))
        jagged = quantile(TreeData([inner_A, inner_B], TreeDim(:study, (:A, :B))), :band => spec; dims=(:draw, :subject))
        @test_throws "ragged trees are not a supported Tables shape" TreeTable(jagged; wide=:band)
        @test_throws "ragged trees are not a supported Tables shape" Tables.columns(TreeTable(jagged))
    end

    # `wide=(:a,:b)` widens several axes at once: the cartesian product of their levels
    # names the columns. Same re-indexing as one dim -- `WideColumn` pins a SET of slots
    # and k == 1 is just NP == 1, not a special case. (1e3figi: treetable.jl's header had
    # promised plural since the pivot landed.)
    @testset "TreeTable(wide=(:a,:b)) pivots several axes at once (1e3figi)" begin
        spec = (lower=0.25, median=0.5, upper=0.75)
        r = quantile(TreeData(reshape(1.0:12.0, 4, 3), :draw, :param), :band => spec; dims=:draw)

        # k == 2 over a 2-dim melt: the row space empties out entirely
        tt2 = TreeTable(r; wide=(:band, :param))
        c2 = Tables.columns(tt2)
        @test length(Tables.columnnames(c2)) == 9          # 3 bands x 3 params, no id columns left
        @test length(Tables.rowtable(tt2)) == 1
        @test :lower_param_1 in Tables.columnnames(c2) && :upper_param_3 in Tables.columnnames(c2)

        # every wide cell still equals the long melt's corresponding cell
        long = Tables.rowtable(TreeTable(r))
        row = only(Tables.rowtable(tt2))
        for lvl in keys(spec), j in 1:3
            @test getproperty(row, Symbol(lvl, "_param_", j)) ==
                  only(filter(l -> l.param == j && l.band === lvl, long)).value
        end

        # `wide` order names the columns; it never changes the DATA
        c2b = Tables.columns(TreeTable(r; wide=(:param, :band)))
        @test collect(Tables.getcolumn(c2b, :param_1_lower)) == collect(Tables.getcolumn(c2, :lower_param_1))

        # widening 2 of 3 axes leaves the third as a real long row axis
        X3 = TreeData(reshape(1.0:24.0, 4, 3, 2), :draw, :param, :arm)
        c3 = Tables.columns(TreeTable(X3; wide=(:param, :arm)))
        @test length(Tables.columnnames(c3)) == 1 + 3*2    # :draw + 6 value columns
        @test length(Tables.getcolumn(c3, :draw)) == 4
        lt = Tables.rowtable(TreeTable(X3))
        for row in Tables.rowtable(TreeTable(X3; wide=(:param, :arm))), p in 1:3, a in 1:2
            @test getproperty(row, Symbol("param_", p, "_arm_", a)) ==
                  only(filter(l -> l.draw == row.draw && l.param == p && l.arm == a, lt)).value
        end

        @test_throws "the same dim more than once" Tables.columns(TreeTable(X3; wide=(:param, :param)))
        @test_throws "not a real axis" Tables.columns(TreeTable(
            TreeData(randn(4), TreeDim(:draw, 1:4), TreeDim(:tag, :fixedtag)); wide=(:tag,)))

        # the multi-slot re-indexing stays allocation-free PER CELL: cost matches a plain
        # Vector and does not grow with rows. (The residual is `@allocated` boxing its own
        # return value -- a plain Vector{Float64} measures exactly the same.)
        sumcol(c) = (t = zero(eltype(c)); for i in eachindex(c); t += c[i]; end; t)
        wcol(n) = Tables.getcolumn(Tables.columns(TreeTable(
            TreeData(reshape(1.0:(n*3*2), n, 3, 2), :draw, :param, :arm); wide=(:param, :arm))), :param_1_arm_2)
        alloc(c) = (sumcol(c); @allocated sumcol(c))
        baseline = alloc(randn(4000))
        @test alloc(wcol(4)) == baseline
        @test alloc(wcol(40_000)) == baseline               # flat in the row count

        # THE headline claim, at k > 1: the pivot re-indexes the long melt's lazy columns,
        # so `columns()` is O(structure), never O(rows). The k=1 gate above did not cover
        # the multi-slot path (`_slotplan` / `_deletemany` / the shared `Val{PLAN}`).
        mk(n) = TreeData(randn(n, 3, 2), :draw, :param=>(:p1,:p2,:p3), :arm=>(:lo,:hi))
        wsmall2 = TreeTable(mk(50);     wide=(:param, :arm))
        wbig2   = TreeTable(mk(50_000); wide=(:param, :arm))
        Tables.columns(wsmall2); Tables.columns(wbig2)      # warm up
        a_small2 = @allocated Tables.columns(wsmall2)
        a_big2   = @allocated Tables.columns(wbig2)
        @info "multi-wide acceptance gate: Tables.columns @allocated (50 vs 50000 draws)" a_small2 a_big2
        @test isapprox(a_big2, a_small2; rtol=0.3)

        # naming at k>1 inherits `_levelname`'s per-dim rule: SYMBOL levels join bare
        # (`p2_hi`), while unlabelled/positional axes stay dim-prefixed (`param_1_arm_2`,
        # asserted above). Mixed dims mix accordingly -- the prefix is per level, not per combo.
        @test length(Tables.getcolumn(Tables.columns(wbig2), :p2_hi)) == 50_000
        @test Set(Tables.columnnames(Tables.columns(wsmall2))) ==
              Set((:draw, :p1_lo, :p2_lo, :p3_lo, :p1_hi, :p2_hi, :p3_hi))
    end

    # Edges of the multi-slot pivot that `35db7ec` left unpinned. `_slotplan` maps each
    # widened slot to its entry of `levels`, so a `wide` order whose slots are NOT
    # ascending is the case most likely to silently transpose data.
    @testset "multi-wide edges: unsorted slots, records, collisions, reps (1e3figi)" begin
        # `wide=(:arm,:param)` => poss = (3,2), non-monotonic. Every cell must still land.
        X3 = TreeData(reshape(1.0:24.0, 4, 3, 2), :draw, :param, :arm)
        lt = Tables.rowtable(TreeTable(X3))
        for row in Tables.rowtable(TreeTable(X3; wide=(:arm, :param))), p in 1:3, a in 1:2
            @test getproperty(row, Symbol("arm_", a, "_param_", p)) ==
                  only(filter(l -> l.draw == row.draw && l.param == p && l.arm == a, lt)).value
        end
        # and the plan really is (reduced, levels[2], levels[1]) -- not (…, 1, 2)
        c = Tables.getcolumn(Tables.columns(TreeTable(X3; wide=(:arm,:param))), :arm_2_param_1)
        @test typeof(c).parameters[end] == (1, -2, -1)

        # a record fans out over BOTH widened axes, each field keeping its own prefix
        rec = TreeData(:rec => (
            a = TreeData(randn(2,2), :band=>(:lo,:hi), :arm=>(:l,:r)),
            b = TreeData(randn(2,2), :band=>(:lo,:hi), :arm=>(:l,:r)),
        ))
        rcols = Tables.columns(TreeTable(rec; wide=(:band, :arm)))
        @test Set(Tables.columnnames(rcols)) == Set((:a_lo_l,:a_hi_l,:a_lo_r,:a_hi_r,
                                                     :b_lo_l,:b_hi_l,:b_lo_r,:b_hi_r))
        rlong = Tables.rowtable(TreeTable(rec))
        rrow  = only(Tables.rowtable(TreeTable(rec; wide=(:band,:arm))))
        rm    = only(filter(l -> l.band === :hi && l.arm === :r, rlong))
        @test rrow.a_hi_r == rm.a && rrow.b_hi_r == rm.b

        # two DISTINCT levels of one dim that sanitize to the same label: blame the DIM,
        # not "another column of the melt" (the per-dim guard the generalization dropped)
        clash = TreeData(randn(4, 2), :draw, :band => (Symbol("q0.025"), :q0_025))
        @test_throws "wide=band has levels that collide" Tables.columns(TreeTable(clash; wide=:band))

        # `_` is not an injective separator: (:x, :x_y) x (:y_z, :z) yields `x_y_z` twice.
        # Loud, never a silently mislabelled column.
        amb = TreeData(randn(2,2,2), :draw, :a => (:x, Symbol("x_y")), :b => (Symbol("y_z"), :z))
        @test_throws "two level combinations join to the same name" Tables.columns(TreeTable(amb; wide=(:a,:b)))

        # a ragged source stays loud under multi-wide, exactly as under k=1 -- a pivot
        # needs ONE level set and ONE column length, and neither survives raggedness.
        # (Long mode melts this tree fine; that is the whole point of long.)
        rag = TreeData([TreeData(randn(n), :time) for n in (2, 3)], :subject)
        @test_throws "not a supported Tables shape under `wide=`" Tables.columns(TreeTable(rag; wide=(:time,)))
        @test length(Tables.rowtable(TreeTable(rag))) == 5

        # the schema stays a STORED (runtime) schema at k>1, with concrete column eltypes
        spec = (lower=0.25, median=0.5, upper=0.75)
        r = quantile(TreeData(reshape(1.0:12.0,4,3), :draw, :param), :band=>spec; dims=:draw)
        sch = Tables.schema(TreeTable(r; wide=(:band,:param)))
        @test sch.names isa Vector{Symbol} && length(sch.names) == 9
        @test all(isconcretetype, sch.types)

        # getindex is fully inferred through the multi-slot reconstruction
        wc = Tables.getcolumn(Tables.columns(TreeTable(r; wide=(:band,:param))), :lower_param_1)
        @test Base.return_types(getindex, (typeof(wc), Int)) == [Float64]

        # both display reps render a multi-wide table (they call `Tables.columns`)
        for m in (MIME"text/html"(), MIME"text/markdown"())
            @test occursin("lower_param_1", sprint(show, m, TreeTable(r; wide=(:band,:param))))
        end
    end

    # Base honours `:limit` for AbstractArrays but NOT for `Tuple` (`show` renders
    # every element). The first cut of html.jl handed tuples straight to `show`, so a
    # 4000-name record axis built a ~128 MB string only to cut it to 60 chars, and a
    # 3000-element TreeTuple leaf emitted 23 KB of uncapped HTML -- both under a
    # header claiming "bounded by construction".
    @testset "display stays bounded on Tuple coords + Tuple leaves" begin
        html(x) = sprint(show, MIME"text/html"(), x)

        tupcoords = TreeData(randn(2), TreeDim(:param, ntuple(i -> Symbol("p", i), 4000)))
        tupleaf   = TreeData(ntuple(i -> Float64(i), 3000), TreeDim(:q, missing))
        arr       = TreeData(randn(1000, 100), :draw, :param)
        html(tupcoords); html(tupleaf); html(arr)        # warm before measuring

        # allocation: was ~128 MB / ~74 MB. A 2 MB bar is ~60x under, so this fails
        # loudly if anyone hands a Tuple to `show` again, without being flaky.
        @test @allocated(html(tupcoords)) < 2_000_000
        @test @allocated(html(tupleaf))   < 2_000_000
        @test @allocated(html(arr))       < 2_000_000

        # emitted HTML is bounded, and every elision is MARKED (never valid-looking)
        @test length(html(tupcoords)) < 1_000 && occursin("…", html(tupcoords))
        @test length(html(tupleaf))   < 1_000 && occursin("3000 elements", html(tupleaf))

        # a short tuple renders in full -- no bogus elision marker
        small = TreeData((1.0, 2.0), TreeDim(:q, (0.1, 0.9)))
        @test occursin("(0.1, 0.9)", html(small)) && !occursin("more)", html(small))

        # the trap the char-clamp alone would miss: 13 SHORT symbols fit under
        # _MAX_COORDS chars, so only the ELEMENT elision can mark the drop.
        shortsyms = TreeData(randn(2), TreeDim(:p, ntuple(i -> Symbol('a' + i - 1), 13)))
        @test occursin("… 5 more", html(shortsyms))
    end

    # ============ the empty / zero-leaf tree (snag: empty-zero-leaf) ============
    # A product-mapped sweep MUST emit a leaf for every cell -- unlike a `reduce(vcat, dfs)`
    # pool, which can simply omit an empty per-combo DataFrame. So "this cell has no data"
    # needs a spelling that (a) reduces, and (b) melts to zero rows.
    #
    # Every structural walk here descends through ONE REPRESENTATIVE child (`first(p)`).
    # An empty boundary has none, and every such site used to throw a bare `BoundsError`.
    # The fix derives structure from the element TYPE instead, which is total.
    @testset "empty / zero-leaf trees" begin
        band = (; median=0.5, lower=0.05, upper=0.95)

        # --- 1. a DENSE leaf with a zero-length labelled axis: already worked, pin it.
        A = TreeData(Float64[], :assay_name => String[])
        @test Tables.schema(A).names == (:assay_name, :value)
        @test isempty(Tables.rowtable(A))

        # --- 2. reduce that dense empty leaf -> TreeArrays produces an EMPTY RAGGED array.
        # Its own melt must consume what its own reduce emits (this was the BoundsError).
        B = nanquantile(TreeData(zeros(100, 0), :draw, :assay_name => String[]), :band => band; dims=:draw)
        @test Tables.schema(B).names == (:assay_name, :band, :value)
        @test isempty(Tables.rowtable(B))

        # --- 3. a zero-length REDUCE dim is NOT a zero-row tree: no draws to summarize
        # means NaN at every band level (NaNStatistics' all-NaN convention), 3 time x 3 band
        # rows -- not zero rows. The two "empties" are different things.
        C = nanquantile(TreeData(zeros(0, 3), :draw, :time => 1:3), :band => band; dims=:draw)
        @test length(Tables.rowtable(C)) == 9
        @test all(isnan, Tables.columntable(C).value)

        # --- 4. the empty ragged nesting, spelled with a CONCRETE element type.
        leafproto = TreeData(zeros(100, 3), :draw, :time => 1:3)
        E = TreeData(typeof(leafproto)[], :assay_name => String[])
        @test TreeArrays._eltype(E) === Float64            # from the TYPE -- no `first` to read
        @test isempty(Tables.rowtable(E))
        @test isempty(Tables.rowtable(nanquantile(E, :band => band; dims=:draw)))

        # --- 5. a product map where EVERY cell is empty -> zero rows, schema intact.
        mkcell(n) = TreeData([TreeData(randn(100, 3), :draw, :time => 1:3) for _ in 1:n],
                             :assay_name => ["a$i" for i in 1:n])
        G = TreeData([mkcell(0), mkcell(0)], :subject => ["s1", "s2"])
        @test Tables.schema(G).names == (:subject, :assay_name, :draw, :time, :value)
        @test isempty(Tables.rowtable(G))
        @test isempty(Tables.rowtable(nanquantile(G, :band => band; dims=:draw)))

        # --- 6. a DENSE zero-row tree reduces and melts to zero rows too (0-length axis
        # in the middle of an otherwise dense array).
        H = nanquantile(TreeData(randn(100, 0, 3), :draw, :subject => String[], :time => 1:3),
                        :band => band; dims=:draw)
        @test Tables.schema(H).names == (:subject, :time, :band, :value)
        @test isempty(Tables.rowtable(H))

        # --- 7. THE BOUNDARY, now MELTABLE. A cell with 0 assays beside a cell with 3 is a
        # RAGGED tree, not a zero-row one: its siblings disagree on shape. No "empty
        # spelling" can make a hole in a RECTANGULAR melt -- but the LONG melt does not need
        # one: the empty sibling simply contributes zero rows, and the row count is the sum.
        # (It used to BoundsError, then to refuse; both are gone.)
        F = TreeData([mkcell(3), mkcell(0)], :subject => ["s1", "s2"])
        @test Tables.schema(F).names == (:subject, :assay_name, :draw, :time, :value)   # type-only: cannot see raggedness
        fcols = Tables.columns(F)
        @test length(Tables.getcolumn(fcols, :value)) == 3 * 100 * 3        # s1's 3 assays; s2 adds nothing
        @test all(==("s1"), Tables.getcolumn(fcols, :subject))             # the empty sibling emits no rows at all
        @test length(Tables.rowtable(nanquantile(F, :band => band; dims=:draw))) == 3 * 3 * 3
        # the same tree under `wide=` still cannot pivot -- one level set, one column length
        @test_throws "not a supported Tables shape under `wide=`" Tables.columns(TreeTable(F; wide=:time))

        # --- 8. reducing the length-0 axis ITSELF has no leaf to push `f` into. Unlike a
        # zero-length NUMERIC slice (case 3), there is no value to invent -- say so.
        @test_throws "cannot reduce a length-0 axis" mapslices(sum, E; dims=:assay_name)

        # --- 9. the abstract spelling the snag reported. It erases the child structure the
        # type walk reads, so BOTH the reduce path and the melt path must reject it by name.
        D = TreeData(TreeData[], :assay_name => String[])
        @test_throws "not a concrete TreeData type" Tables.schema(D)
        @test_throws "not a concrete TreeData type" TreeArrays._eltype(D)
    end

# an index-backed outer axis: each leaf is BUILT on `getindex`, so indexing it is observable.
struct _LazyLeaves{T,F} <: AbstractVector{T}
    n::Int
    f::F
end
_LazyLeaves(n, f) = _LazyLeaves{typeof(f(1)),typeof(f)}(n, f)
Base.size(L::_LazyLeaves) = (L.n,)
Base.IndexStyle(::Type{<:_LazyLeaves}) = IndexLinear()
Base.getindex(L::_LazyLeaves, i::Int) = L.f(i)

    @testset "outer-axis reduce indexes a lazy parent once per leaf" begin
        # `TreeRaggedArray`'s `P<:AbstractArray{<:TreeData}` admits an index-backed parent that
        # builds its leaves on `getindex`. An outer-axis reduce must walk that parent ONCE per
        # leaf: an inner-major gather rebuilds each leaf `length(leaf)` times, which is silent
        # (correct values, no error) and scales with exactly the size that motivates laziness.
        builds = Ref(0)
        Xl = TreeData(_LazyLeaves(4, i -> (builds[] += 1; TreeData(fill(Float64(i), 25), :row))), :boot)
        @test Xl isa TreeRaggedArray

        builds[] = 0
        r = mapslices(sum, Xl; dims=:boot)
        @test builds[] == 4                                   # == n_boot, NOT 1 + 25*4
        @test parent(r) == fill(Float64(sum(1:4)), 25)        # and the values are right

        # the count must not track the INNER length -- that is the regression that bit Bruno.
        for n_inner in (25, 50, 100)
            Xn = TreeData(_LazyLeaves(4, i -> (builds[] += 1; TreeData(fill(Float64(i), n_inner), :row))), :boot)
            builds[] = 0
            mapslices(sum, Xn; dims=:boot)
            @test builds[] == 4
        end

        # an eager parent must be untouched by the same code path
        Xe = TreeData([TreeData(fill(Float64(i), 25), :row) for i in 1:4], :boot)
        @test parent(mapslices(sum, Xe; dims=:boot)) == parent(r)
    end

    # Bruno asked whether `quantile`/`nanquantile`'s `dims=` takes a COLLECTION, to pool
    # draws AND subjects in one pass (the `combine(groupby(df, [:dose, :study]), :qoi =>
    # quantile)` shape). It always has -- `_dimnames(dims) = Tuple(dims)` -- but NOTHING
    # pinned it, which is exactly why the capability's status was unknowable from outside.
    # Pin the contract: a multi-dim reduce is JOINT/POOLED (one bag, one sorted pass), not
    # the composition of two reductions.
    @testset "dims= accepts a collection: joint pooled reduce over several axes" begin
        ps = (0.025, 0.25, 0.5, 0.75, 0.975)
        A  = randn(40, 6, 3)
        X  = TreeData(A, :draw, :subject, :time => 1:3)

        # the kernel sees the whole (draw x subject) block, not a vector
        shp = mapslices(sl -> (ndims(sl), size(sl)), X; dims=(:draw, :subject))
        @test parent(shp)[1] == (2, (40, 6))

        # pooled == base-Julia quantile of the flattened block, EXACTLY
        r = quantile(X, TreeDim(:ribbon, ps); dims=(:draw, :subject))
        for t in 1:3
            @test collect(parent(parent(r)[t])) == quantile(vec(A[:, :, t]), collect(ps))
        end
        # ... and pooling is NOT quantile-of-quantiles: the two genuinely differ
        seq = [quantile([quantile(A[:, s, t], 0.5) for s in 1:6], 0.5) for t in 1:3]
        @test seq != [quantile(vec(A[:, :, t]), 0.5) for t in 1:3]

        @test all(parent(mean(X; dims=(:draw, :subject)))[t] ≈ mean(vec(A[:, :, t])) for t in 1:3)
        @test all(parent(sum(X; dims=(:draw, :subject)))[t] ≈ sum(vec(A[:, :, t])) for t in 1:3)

        # a Vector of names works as well as a Tuple; reducing EVERY axis yields one leaf
        rv = quantile(X, TreeDim(:ribbon, ps); dims=[:draw, :subject])
        @test collect(parent(parent(rv)[1])) == collect(parent(parent(r)[1]))
        @test collect(parent(quantile(X, TreeDim(:ribbon, ps); dims=(:draw, :subject, :time)))) ==
              quantile(vec(A), collect(ps))

        # reduced dims survive as ghosts, in dim order; kept axes stay live
        @test map(TreeArrays.name, TreeArrays.dims(r)) == (:time, :draw, :subject)

        # NaN-safe twin: same pooling, NaNs dropped from the pooled bag
        B = copy(A); B[1, 1, 1] = NaN
        rn = nanquantile(TreeData(B, :draw, :subject, :time => 1:3), TreeDim(:ribbon, ps);
                         dims=(:draw, :subject))
        for t in 1:3
            @test collect(parent(parent(rn)[t])) == quantile(filter(!isnan, vec(B[:, :, t])), collect(ps))
        end

        # both wide forms accept the collection too
        rw = nanquantile(X, (median=0.5, lo=0.025); dims=(:draw, :subject))
        @test parent(parent(rw)[1]).median ≈ quantile(vec(A[:, :, 1]), 0.5)
        rp = quantile(X, :band => (lo=0.025, med=0.5); dims=(:draw, :subject))
        @test parent(parent(rp)[1])[2] ≈ quantile(vec(A[:, :, 1]), 0.5)

        # a multi-dim reduce composes with a prior reduction's ghost dims
        C  = randn(12, 5, 4, 2)
        Xc = TreeData(C, :draw, :subject, :time => 1:4, :dose => [1, 2])
        r2 = quantile(mapslices(maximum, Xc; dims=:time), TreeDim(:ribbon, ps); dims=(:draw, :subject))
        qoi = [maximum(C[d, s, :, k]) for d in 1:12, s in 1:5, k in 1:2]
        for k in 1:2
            @test collect(parent(parent(r2)[k])) == quantile(vec(qoi[:, :, k]), collect(ps))
        end
    end

    # The gather-loop indexes every leaf with the FIRST leaf's CartesianIndices, and
    # `_reduceouter` stops once it has reduced this node's outer axis. Both silently produced
    # a wrong answer; both must now name the problem instead.
    @testset "ragged reduces refuse the shapes they cannot serve (never silently wrong)" begin
        mat  = reshape(collect(1.0:36.0), 3, 12)
        idx  = [1:3, 4:8, 9:12]          # subject time-lengths 3, 5, 4 -- genuinely ragged
        subj = TreeData(map(s -> TreeData(view(mat, :, idx[s]), :draw, :time), 1:3), :subject)

        # (a) non-conformable leaves: the gather would read the wrong cells of leaves 2,3 and
        #     drop their tails. Previously returned a plausible 3x3 matrix with no error.
        @test_throws "not conformable" mean(subj; dims=:subject)

        # (b) `dims=` straddling the ragged boundary while a KEPT inner axis is itself ragged
        #     (here :time is kept and has lengths 3/5/4): the pooled result cannot be rectangular,
        #     so the conformability check on the kept axes refuses it by name.
        @test_throws "not conformable" mean(subj; dims=(:draw, :subject))
        @test_throws "not conformable" quantile(subj, TreeDim(:r, (0.5,)); dims=(:draw, :subject))

        # the LEGAL chained flow (skill §8) is untouched: collapse the ragged inner axis, THEN
        # reduce the outer one.
        step1 = mapslices(maximum, subj; dims=:time)   # -> (subject -> (draw,)), conformable
        @test collect(parent(mean(step1; dims=:subject))) ≈
              [mean([maximum(mat[d, idx[s]]) for s in 1:3]) for d in 1:3]

        # ...and once the kept axes ARE conformable, a straddle now POOLS across the ragged
        # boundary (snag pooled-reduce-ov): pool :draw AND :subject into one bag -- identical to
        # the dense equivalent, NOT the composition of two reductions. (Was refused pre-a4734df9.)
        @test parent(mean(step1; dims=(:draw, :subject))) ≈
              mean([maximum(mat[d, idx[s]]) for d in 1:3, s in 1:3])

        # a purely-inner multi-dim reduce never reaches the straddle path
        wide = TreeData([TreeData(randn(3, 4, 2), :draw, :time, :chan) for _ in 1:3], :subject)
        @test mapslices(mean, wide; dims=(:time, :chan)) isa TreeData
    end

    # Pooled reduce that STRADDLES a ragged nesting boundary (snag pooled-reduce-ov-a4734df9,
    # reporter Bruno:treearrays): `dims=(:draw,:chain)` on a `(chain -> (draw,param))` ragged tree
    # pools ALL draws x chains per param -- the shape a pooled posterior CI needs from per-chain
    # @mmap'd matrices held as a lazy collection (never hcat'd into a dense block). The anchor: it
    # must equal, byte-for-byte, the DENSE `TreeData(A, :draw,:chain,:param)` reduce (which the
    # pooled-reduce testset above already pins against base-Julia).
    @testset "pooled reduce across a ragged nesting boundary (dims straddle)" begin
        ps = (0.025, 0.5, 0.975)
        nchain, ndraw, nparam = 4, 40, 3
        A = randn(ndraw, nchain, nparam)
        dense  = TreeData(A, :draw, :chain, :param => (:a, :b, :c))
        ragged = TreeData([TreeData(A[:, c, :], :draw, :param => (:a, :b, :c)) for c in 1:nchain], :chain)
        @test ragged isa TreeArrays.TreeRaggedArray

        r_ragged = quantile(ragged, TreeDim(:ribbon, ps); dims=(:draw, :chain))
        r_dense  = quantile(dense,  TreeDim(:ribbon, ps); dims=(:draw, :chain))
        for j in 1:nparam   # byte-identical to the dense reduce AND to the base-Julia pool
            @test collect(parent(parent(r_ragged)[j])) == collect(parent(parent(r_dense)[j]))
            @test collect(parent(parent(r_ragged)[j])) == quantile(vec(A[:, :, j]), collect(ps))
        end
        # kept :param live; :draw + :chain sliced ghosts (same dim SET as the dense reduce)
        @test map(TreeArrays.name, TreeArrays.dims(r_ragged)) == (:param, :draw, :chain)

        # mean/sum + both wide spellings pool identically; reducing EVERY axis -> one leaf
        @test all(parent(mean(ragged; dims=(:draw, :chain)))[j] ≈ mean(vec(A[:, :, j])) for j in 1:nparam)
        @test all(parent(sum(ragged;  dims=(:draw, :chain)))[j] ≈ sum(vec(A[:, :, j]))  for j in 1:nparam)
        # std/var pool the SAME bag (snag std-var-pooled-r, reporter Bruno:treearrays): byte-identical
        # to the DENSE reduce AND to base-Julia over the flattened (draw x chain) block. `corrected`
        # defaults to `true` => SAMPLE variance (÷(n-1)), matching `std(some_vector)`; `false` => ÷n.
        for (r_ragged, base) in ((std(ragged; dims=(:draw,:chain)), std), (var(ragged; dims=(:draw,:chain)), var))
            @test all(parent(r_ragged)[j] == base(vec(A[:, :, j])) for j in 1:nparam)                     # == base-Julia
        end
        @test all(parent(std(ragged; dims=(:draw,:chain)))[j] == parent(std(TreeData(A, :draw,:chain,:param=>(:a,:b,:c)); dims=(:draw,:chain)))[j] for j in 1:nparam)  # == dense
        n = ndraw * nchain
        @test all(parent(var(ragged; dims=(:draw,:chain), corrected=false))[j] == var(vec(A[:,:,j]); corrected=false) for j in 1:nparam)   # population (÷n)
        @test parent(var(ragged; dims=(:draw,:chain)))[1] / parent(var(ragged; dims=(:draw,:chain), corrected=false))[1] ≈ n / (n - 1)      # Bessel factor
        rp = quantile(ragged, :band => (lo=0.025, med=0.5); dims=(:draw, :chain))
        @test parent(parent(rp)[1])[2] ≈ quantile(vec(A[:, :, 1]), 0.5)
        rw = nanquantile(ragged, (median=0.5, lo=0.025); dims=(:draw, :chain))
        @test parent(parent(rw)[1]).median ≈ quantile(vec(A[:, :, 1]), 0.5)
        @test collect(parent(quantile(ragged, TreeDim(:ribbon, ps); dims=(:draw, :chain, :param)))) ==
              quantile(vec(A), collect(ps))

        # the per-chain backing matrices are REFERENCED, not copied into a pooled block
        @test parent(parent(ragged)[1]) === parent(parent(ragged)[1])

        # KEPT outer axis (a 2-D grid of ragged cells: pool :chain, keep :grp) is still
        # unimplemented -> throws by name rather than answer a partial pool
        grid = TreeData([TreeData(randn(5, 3), :draw, :param) for _ in 1:6, _ in 1:2], :chain, :grp)
        @test_throws "KEEPING part of this node's outer axis" mean(grid; dims=(:draw, :chain))
    end


    # `dims=` is foundALL, not foundany (decision 1iy1r57, user-directed). A name that resolves
    # NOWHERE used to reduce nothing and yield the `missing` sentinel -- a typo'd dim produced a
    # plausible-looking result rather than an error, which is exactly what 16fwcnx forbids. The
    # check is type-level and `@generated`, so a correct `dims=` costs nothing at runtime.
    @testset "dims= is foundALL: a name that resolves nowhere throws (1iy1r57)" begin
        X = TreeData(reshape(1.0:12.0, 4, 3), :draw, :param)

        @test_throws "exists nowhere in this tree" mapslices(sum, X; dims=:drwa)
        # a typo INSIDE a collection: its siblings used to reduce, so the result looked reduced
        @test_throws "exists nowhere in this tree" mapslices(sum, X; dims=(:draw, :prm))
        @test_throws "Available dims: (:draw, :param)" mapslices(sum, X; dims=:nope)

        # legitimate reduces are untouched
        @test parent(mapslices(sum, X; dims=:draw)) == [10.0, 26.0, 42.0]
        @test mapslices(sum, X; dims=(:draw, :param)) |> parent == 78.0

        # a dim living on a LEAF is found from the root -- not mistaken for a typo
        rag = TreeData([TreeData(randn(4, 3), :draw, :param) for _ in 1:3], :subject)
        @test mapslices(mean, rag; dims=:draw) isa TreeData
        nt = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(4), :draw)))
        @test keys(parent(mapslices(mean, nt; dims=:draw))) == (:a, :b)

        # A JAGGED tree hides its children's names behind a non-concrete eltype. A legitimate
        # reduce must still work...
        P = reshape(1.0:42.0, 6, 7)
        iA = TreeData([TreeData(view(P,:,[1,2]), :draw, :subject), TreeData(view(P,:,[3]), :draw, :subject)],
                      TreeDim(:dose, (10, 20)))
        iB = TreeData([TreeData(view(P,:,[4,5]), :draw, :subject)], TreeDim(:dose, (20,)))
        jag = TreeData([iA, iB], TreeDim(:study, (:A, :B)))
        @test quantile(jag, :band => (lo=0.25, hi=0.75); dims=(:draw, :subject)) isa TreeData

        # ...and the assert must NOT stand down there. An earlier cut skipped the check whenever
        # the type walk was incomplete, on the reasoning that absence could not be *proven*. That
        # left a silent wrong answer: `dims=(:draw,:drwa)` reduced `:draw`, dropped the typo, and
        # returned a plausible result. Absence is always provable -- just not always from the type.
        @test_throws "exists nowhere in this tree" mapslices(sum, jag; dims=(:draw, :drwa))
        @test_throws "exists nowhere in this tree" mapslices(sum, jag; dims=:drwa)
        # and a dim that lives BELOW the jagged boundary is still found, not called a typo
        @test mapslices(sum, jag; dims=:subject) isa TreeData

        # the `missing` sentinel is RETIRED as a reduction output: a branch that lacks a dim the
        # tree has elsewhere is a heterogeneous-dims shape, which the Tables adapter already
        # refuses. It now says so at the source instead of travelling to a melt that rejects it.
        het = TreeData(:rec=>(;a=TreeData(randn(4), :draw), b=TreeData(randn(3), :other)))
        @test_throws "heterogeneous-dims shape" mapslices(mean, het; dims=:draw)

        # `missing` is still the UNLABELLED-AXIS marker -- a different job, untouched (see below)
        @test TreeArrays._isaxis(TreeDim(:draw))
        @test TreeArrays.meta(TreeDim(:draw)).values === missing
    end

    @testset "TreeActualArray — opt-in lazy AbstractArray view (a967zd)" begin
        nd, nc, np = 50, 4, 3
        arr = reshape(collect(1.0:(nd*nc*np)), nd, nc, np)
        pnames = (:alpha, :beta, :gamma)

        # rep 1 — dense leaf (parent IS the array); rep 2 — RECORD of per-param (draw,chain)
        # matrices (parent is a NamedTuple, NOT an array) — same underlying data.
        X1 = TreeData(arr, :draw, :chain, :param => collect(pnames))
        A1 = TreeActualArray(X1)
        fields = NamedTuple{pnames}(ntuple(k -> arr[:, :, k], np))
        X2 = TreeData(:param => fields, :draw, :chain)
        A2 = TreeActualArray(X2)

        @test A1 isa AbstractArray{Float64,3}
        @test A2 isa AbstractArray{Float64,3}
        @test !(X1 isa AbstractArray)                # the TreeData itself stays not-an-AbstractArray
        @test size(A1) == (nd, nc, np) == size(A2)
        @test collect(A1) == arr                     # both representations reconstruct the SAME array
        @test collect(A2) == arr
        @test A1 == A2
        @test sum(A2; dims = 1) == sum(arr; dims = 1) # acts like any other AbstractArray
        @test @inferred(A1[1, 1, 1]) === arr[1, 1, 1] # type-stable getindex
        @test @inferred(A2[1, 1, 1]) === arr[1, 1, 1]
        @test TreeArrays.name.(dims(A1)) == (:draw, :chain, :param)   # labels retained into array-land
        @test parent(A2) === X2

        # zero-copy: a record of VIEWS reflects source mutation (nothing materialized)
        A2v = TreeActualArray(TreeData(:param => NamedTuple{pnames}(ntuple(k -> view(arr, :, :, k), np)), :draw, :chain))
        arr[2, 2, 2] = -777.0
        @test A1[2, 2, 2] == -777.0
        @test A2v[2, 2, 2] == -777.0

        # ess/rhat signature (real MCMCDiagnosticTools: `AbstractArray{<:Union{Missing,Real}}`)
        ess_like(x::AbstractArray{<:Union{Missing,Real}}) = size(x)
        @test ess_like(A1) == (nd, nc, np)
        @test ess_like(A2) == (nd, nc, np)

        # rectangular only — ragged / heterogeneous shape / heterogeneous eltype each error
        @test_throws ErrorException TreeActualArray(TreeData([TreeData(randn(3), :t), TreeData(randn(4), :t)], :s))
        @test_throws ErrorException TreeActualArray(TreeData(:param => (a = randn(5, 4), b = randn(6, 4)), :draw, :chain))
        @test_throws ErrorException TreeActualArray(TreeData(:param => (a = randn(5, 4), b = rand(1:9, 5, 4)), :draw, :chain))

        # unequal per-chain :draw counts error BY NAME — outer axis + differing inner
        # axis + lengths (snag treeactualarray-998ba536: TA must never hand back an
        # array for ragged chains, and the refusal must say so precisely)
        rag_uneq = TreeData([TreeData(randn(n, 2), :draw, :param => [:a, :b]) for n in (100, 87)], :chain)
        msg_uneq = try TreeActualArray(rag_uneq); nothing catch e e.msg end
        @test msg_uneq !== nothing
        @test occursin("`:chain`", msg_uneq) && occursin("`:draw`", msg_uneq)
        @test occursin("[100, 87]", msg_uneq)

        # an outer axis of sub-trees is refused EVEN WHEN rectangular — slicing to a
        # common length does not unblock TreeActualArray (no ragged→dense stacker exists)
        rag_eq = TreeData([TreeData(randn(100, 2), :draw, :param => [:a, :b]) for _ in 1:2], :chain)
        msg_eq = try TreeActualArray(rag_eq); nothing catch e e.msg end
        @test msg_eq !== nothing
        @test occursin("`:chain`", msg_eq) && occursin("even when rectangular", msg_eq)
    end

    @testset "coords=true — a kernel sees the axis it reduces (snag kernels-cannot-s)" begin
        # AUC and tmax: the two standard non-compartmental PK summaries. Both need `t`, and the
        # positional workarounds (`v[1]`, `v .- v[1]`) provably cannot express either.
        trapz(t, y) = sum(i -> (t[i+1] - t[i]) * (y[i+1] + y[i]) / 2, 1:length(t)-1)
        rec(r, i) = parent(parent(r)[i])   # a kept axis leaves `parent` an array of leaves

        ts = [0.0, 0.5, 2.0, 6.0]
        Y  = [1.0 2.0; 4.0 1.0; 2.0 3.0; 0.5 8.0]        # (time=4, subject=2)
        X  = TreeData(Y, :time => ts, :subject => [:a, :b])

        r = mapslices(X; dims = :time, coords = true) do y, t
            TreeData(:stat => (; cmax = maximum(y), tmax = t[argmax(y)], auc = trapz(t, y)))
        end
        @test rec(r, 1).tmax == ts[argmax(Y[:, 1])] == 0.5
        @test rec(r, 2).tmax == ts[argmax(Y[:, 2])] == 6.0
        @test rec(r, 1).auc ≈ trapz(ts, Y[:, 1]) ≈ 10.75
        @test rec(r, 2).auc ≈ trapz(ts, Y[:, 2]) ≈ 25.75
        @test rec(r, 1).auc != rec(r, 2).auc              # the two subjects really do differ
        # structure is byte-identical to the 1-arg path: the reduced dim stays as a ghost
        @test TreeArrays.name.(dims(r)) == (:subject, :time)
        @test TreeArrays.meta(dims(r)[2]).values === nothing
        @test rec(mapslices(y -> TreeData(:stat => (; cmax = maximum(y))), X; dims = :time), 1).cmax ==
              rec(r, 1).cmax

        # the DEFAULT is unchanged, and coords=false is explicitly today's behaviour
        @test parent(mapslices(maximum, X; dims = :time)) == [4.0, 8.0]
        @test parent(mapslices(maximum, X; dims = :time, coords = false)) == [4.0, 8.0]

        # RAGGED — the case the "close over a shared constant" escape provably cannot serve:
        # the coordinates DIFFER per sub-tree, and each kernel call must see its OWN grid.
        t1, t2 = [0.0, 1.0, 3.0], [0.0, 0.5, 2.0, 8.0]
        y1, y2 = [1.0, 5.0, 2.0], [2.0, 3.0, 9.0, 1.0]
        rag = TreeData([TreeData(y1, :time => t1), TreeData(y2, :time => t2)], :subject => [:s1, :s2])
        rr = mapslices(rag; dims = :time, coords = true) do y, t
            TreeData(:stat => (; tmax = t[argmax(y)], auc = trapz(t, y), n = length(t)))
        end
        @test parent(parent(rr)[1]).tmax == 1.0           # s1's grid
        @test parent(parent(rr)[2]).tmax == 2.0           # s2's DIFFERENT grid
        @test parent(parent(rr)[1]).auc ≈ trapz(t1, y1)
        @test parent(parent(rr)[2]).auc ≈ trapz(t2, y2)
        @test (parent(parent(rr)[1]).n, parent(parent(rr)[2]).n) == (3, 4)   # lengths really differ

        # reducing the ragged OUTER axis hands the kernel THAT axis's labels
        conf = TreeData([TreeData([1.0, 2.0, 3.0], :time => ts[1:3]),
                         TreeData([4.0, 6.0, 8.0], :time => ts[1:3])], :subject => [:a, :b])
        @test parent(parent(mapslices((y, c) -> (@test c == [:a, :b]; sum(y)), conf;
                                      dims = :subject, coords = true))) == [5.0, 8.0, 11.0]

        # SEVERAL reduced axes -> one coordinate vector per axis, in slice-dim order
        A  = collect(reshape(1.0:24.0, 4, 3, 2))
        X3 = TreeData(A, :draw => 1:4, :chain => [:c1, :c2, :c3], :param => [:p, :q])
        r3 = mapslices(X3; dims = (:draw, :chain), coords = true) do s, c
            @test c isa Tuple && length(c) == 2 && c[1] == 1:4 && c[2] == [:c1, :c2, :c3]
            @test size(s) == (4, 3)
            sum(s)
        end
        @test parent(r3) == [sum(A[:, :, 1]), sum(A[:, :, 2])]

        # an UNLABELLED axis has no coordinates: error BY NAME, never pass `missing` through
        # (that would be the retired absent-dim sentinel in a new costume).
        Xu = TreeData(Y, :time, :subject => [:a, :b])
        @test_throws "unlabelled" mapslices((y, t) -> maximum(y), Xu; dims = :time, coords = true)
        @test_throws "no coordinates" mapslices((y, t) -> maximum(y), Xu; dims = :time, coords = true)
        @test parent(mapslices(maximum, Xu; dims = :time)) == [4.0, 8.0]   # 1-arg still fine

        # a POOLED straddle has no single aligned coordinate vector -> refuse by name
        pooled = TreeData([TreeData(randn(5, 2), :draw => 1:5, :param => [:p, :q]) for _ in 1:3],
                          :chain => 1:3)
        @test mean(pooled; dims = (:draw, :chain)) isa TreeData            # the 1-arg pool works
        @test_throws "POOLING" mapslices((s, c) -> mean(s), pooled; dims = (:draw, :chain), coords = true)

        # type stability: the literal `coords=` kwarg folds, both dense and ragged
        auc(y, t) = trapz(t, y)
        f_dense(Z) = mapslices(auc, Z; dims = :time, coords = true)
        g_dense(Z) = mapslices(maximum, Z; dims = :time)
        f_rag(Z) = mapslices(auc, Z; dims = :time, coords = true)
        @test @inferred(f_dense(X)) isa TreeData
        @test @inferred(g_dense(X)) isa TreeData
        @test @inferred(f_rag(rag)) isa TreeData
        @test parent(@inferred(f_dense(X))) ≈ [trapz(ts, Y[:, 1]), trapz(ts, Y[:, 2])]
    end

    @testset "coords(d) / coords(X, :dim) — a CONSUMER can read an axis's labels too" begin
        # Reported alongside the kernel case: the coordinates were unreachable from BOTH
        # directions. `TreeDim`'s only field is `meta`, so `d.values` is a plain getfield
        # failure, and `meta`/`name` are internal.
        ts = [0.0, 0.5, 2.0]
        X  = TreeData([1.0 2.0; 4.0 1.0; 2.0 3.0], :time => ts, :subject => [:a, :b]; dose = 20)
        @test fieldnames(TreeDim) == (:meta,)                       # why `d.values` cannot work
        @test_throws ErrorException coords(X, :nope)                # foundALL, like dims=

        @test coords(dims(X)[1]) === ts                             # zero-copy, exact container
        @test coords(X, :time) === ts
        @test coords(X, :subject) == [:a, :b]
        # inferrable — but note WHERE the literal has to be. A runtime `Symbol` would make the
        # return type the UNION of every dim's coordinate type on a heterogeneous tree, so the
        # name is staged (`Val`) behind `@constprop`; the literal must therefore sit in the
        # CODE, exactly as for `dims=` (§5). `@inferred` works from argument TYPES, so it
        # cannot see const-prop through its own call — hence the wrapper, not a bare @inferred.
        _ctime(Z) = coords(Z, :time)
        _csubj(Z) = coords(Z, :subject)
        @test @inferred(_ctime(X)) === ts
        @test @inferred(_csubj(X)) == [:a, :b]
        @test @inferred(coords(dims(X)[1])) === ts                  # 1-arg form: always stable

        # keep-as-provided, like quantile's levels / selectdim's labels
        @test coords(TreeData(randn(3), :t => (1, 2, 3)), :t) === (1, 2, 3)
        @test coords(TreeData(randn(3), :t => 1:3), :t) === 1:3

        # `collect(d)` is the TRAP this accessor replaces: it runs (TreeDim has iterate/length)
        # but yields Any-eltype, and answers a plausible-looking length-1 vector for the two
        # kinds that have NO coordinates. Pin both halves so the contrast can't silently rot.
        du = dims(TreeData(randn(3, 2), :time, :subject => [:a, :b]))[1]
        # `isequal`, not `==`: `[missing] == [missing]` is itself `missing` (part of the trap)
        @test isequal(collect(du), [missing]) && eltype(collect(du)) === Any
        @test_throws "unlabelled" coords(du)                             # the accessor refuses
        red = mapslices(maximum, X; dims = :time)
        @test collect(dims(red)[2]) == [nothing]                         # the trap again
        @test_throws "REDUCED away" coords(red, :time)                   # ghost: gone by construction
        @test_throws "fixed single position" coords(X, :dose)            # scalar: not an axis
        @test coords(X, :time) === ts                                    # ...and reducing didn't
        @test coords(red, :subject) == [:a, :b]                          # touch the kept axis

        # ragged: each child carries its OWN :time coords, and the message says where to look
        rag = TreeData([TreeData(randn(3), :time => ts), TreeData(randn(4), :time => 1.0:4.0)],
                       :subject => [:s1, :s2])
        @test coords(rag, :subject) == [:s1, :s2]
        @test_throws "on the child" coords(rag, :time)
        @test coords(parent(rag)[1], :time) === ts
        @test coords(parent(rag)[2], :time) === 1.0:4.0
    end

    @testset "@kernel infers the coords opt-in from arity (snag kernels-cannot-s)" begin
        # arity-sniffing is safe HERE because the macro reads the literal argument list, not a
        # value: at the `mapslices` boundary `hasmethod(maximum, (y, t))` is TRUE (`maximum(f,
        # itr)`), so an arity probe there would treat the data slice as a predicate.
        @test hasmethod(maximum, Tuple{Vector{Float64},Vector{Float64}})   # why not at mapslices

        t1, t2 = [0.0, 1.0, 3.0], [0.0, 0.5, 2.0, 8.0]
        rag = TreeData([TreeData([1.0, 5.0, 2.0], :time => t1),
                        TreeData([2.0, 3.0, 9.0, 1.0], :time => t2)], :subject => [:s1, :s2])
        X = TreeData([1.0 2.0; 4.0 1.0; 2.0 3.0], :time => t1, :subject => [:a, :b])
        rec(r, i) = parent(parent(r)[i])

        # 2 plain positional args -> coords threaded through automatically
        @test parent(parent(_tt_nca(rag))[1]).tmax == 1.0     # s1's own grid
        @test parent(parent(_tt_nca(rag))[2]).tmax == 2.0     # s2's own grid
        @test _tt_nca([1.0, 5.0, 2.0], t1).tmax == 1.0        # plain-array kernel still emitted
        @test rec(_tt_nca(X), 1).tmax == 1.0

        # 1 plain positional arg -> today's behaviour, byte-identical
        @test rec(_tt_compute_stats(X), 1).peak == 4.0

        # a DEFAULT and a SPLAT are 1-slice kernels with extra machinery, NOT coordinate
        # consumers -- auto-passing coordinates into their second slot would corrupt them.
        @test rec(_tt_withdefault(X), 1).peak == 8.0
        @test rec(_tt_withsplat(X), 1).peak == 4.0

        # post hoc has no argument list to read: defaults to 1-arg, takes an explicit opt-in
        @test rec(_tt_posthoc_plain(X), 1).peak == 4.0
        @test parent(parent(_tt_posthoc_coords(rag))[1]).tmax == 1.0

        # contradictions and unsupported arities are macro-expansion errors, by name
        @test_throws Exception @eval @kernel (:time => :stat) coords=true _bad1(L) = maximum(L)
        @test_throws Exception @eval @kernel (:time => :stat) coords=false _bad2(L, t) = maximum(L)
        @test_throws Exception @eval @kernel (:time => :stat) coords=yes _bad3(L) = maximum(L)
        @test_throws Exception @eval @kernel (:time => :stat) _bad4(L, t, z) = maximum(L)
    end

end
