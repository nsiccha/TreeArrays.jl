# FlexiChains integration tests (TreeArraysFlexiChainsExt). Order-sensitive
# assertions use 3D-ctor chains (tuple order is preserved); `Dict`-built chains
# do NOT preserve insertion order, so those compare as sets.
using Test
using TreeArrays
using TreeArrays: name
using FlexiChains
using FlexiChains: FlexiChain, Parameter, Extra, VarName, @varname,
    parameters, extras, niters, nchains, iter_indices, chain_indices,
    stat_indices, collapse

@testset "FlexiChains ext" begin

    @testset "ingest scalar SymChain" begin
        arr = reshape(Float64.(1:60), 10, 2, 3)
        chain = FlexiChain{Symbol}(arr, (Parameter(:x), Parameter(:y), Extra(:lp)))
        X = TreeData(chain)
        @test name(outerdim(X)) == :param
        @test map(name, TreeArrays.dims(X)) == (:draw, :chain, :param)
        @test sort(collect(propertynames(X))) == [:x, :y]   # extras excluded by default
        @test coords(X, :draw) == collect(1:10)
        @test coords(X, :chain) == collect(1:2)
        @test parent(X).x == arr[:, :, 1]
        @test parent(X).y == arr[:, :, 2]
        # property access + reductions work on the ingested tree
        @test parent(X.x) == arr[:, :, 1]
        R = mean(X; dims=:draw)
        @test vec(parent(parent(R).x)) == vec(mean(arr[:, :, 1]; dims=1))
        @test vec(parent(parent(R).y)) == vec(mean(arr[:, :, 2]; dims=1))
    end

    @testset "ingest extras opt-in + offset indices" begin
        mx = reshape(Float64.(1:20), 10, 2)
        mlp = reshape(Float64.(101:120), 10, 2)
        chain = FlexiChain{Symbol}(10, 2,
            Dict(Parameter(:x) => mx, Extra(:lp) => mlp); iter_indices=101:110)
        X = TreeData(chain)
        @test collect(propertynames(X)) == [:x]
        Xe = TreeData(chain; extras=true)
        @test sort(collect(propertynames(Xe))) == [:lp, :x]
        @test parent(Xe).lp == mlp
        @test coords(Xe, :draw) == collect(101:110)   # sampler offsets survive
        @test coords(Xe, :chain) == collect(1:2)
        # a Parameter/Extra collision throws rather than shadowing
        bad = FlexiChain{Symbol}(10, 2,
            Dict(Parameter(:a) => mx, Extra(:a) => mlp))
        @test TreeData(bad) isa TreeData                      # params-only is unambiguous
        @test_throws ArgumentError TreeData(bad; extras=true)
    end

    @testset "ingest VarName chain" begin
        mx = reshape(Float64.(1:20), 10, 2)
        my1 = reshape(Float64.(21:40), 10, 2)
        my2 = reshape(Float64.(41:60), 10, 2)
        vchain = FlexiChain{VarName}(10, 2, Dict(
            Parameter(@varname(x)) => mx,
            Parameter(@varname(y[1])) => my1,
            Parameter(@varname(y[2])) => my2))
        X = TreeData(vchain)
        @test Set(propertynames(X)) == Set([:x, Symbol("y[1]"), Symbol("y[2]")])
        @test parent(X).x == mx
        @test parent(X)[Symbol("y[1]")] == my1
        @test parent(X)[Symbol("y[2]")] == my2
        R = mean(X; dims=:draw)
        @test vec(parent(parent(R)[Symbol("y[2]")])) == vec(mean(my2; dims=1))
    end

    @testset "ingest array-valued keys nest" begin
        mx = reshape(Float64.(1:20), 10, 2)
        mz = [Float64[100i + j, 200i + j] for i in 1:10, j in 1:2]
        chain = FlexiChain{Symbol}(10, 2, Dict(Parameter(:x) => mx, Parameter(:z) => mz))
        X = TreeData(chain)
        z = parent(X).z
        @test z isa TreeData
        @test map(name, TreeArrays.dims(z)) == (:draw, :chain, :elem)
        @test size(parent(z)) == (10, 2, 2)
        @test parent(z)[3, 2, 1] == mz[3, 2][1]
        @test parent(z)[3, 2, 2] == mz[3, 2][2]
        @test coords(z, :draw) == collect(1:10)
        # matrix-valued cells nest with one axis per inner dim
        mm = [fill(Float64(10i + j), 2, 2) for i in 1:10, j in 1:2]
        mchain = FlexiChain{Symbol}(10, 2, Dict(Parameter(:m) => mm))
        m = parent(TreeData(mchain)).m
        @test map(name, TreeArrays.dims(m)) == (:draw, :chain, :elem_1, :elem_2)
        @test size(parent(m)) == (10, 2, 2, 2)
        # ragged / mixed / non-numeric cells throw -- nothing silently dropped
        rag = [i == 1 ? [1.0] : [1.0, 2.0] for i in 1:10, j in 1:2]
        ragchain = FlexiChain{Symbol}(10, 2, Dict(Parameter(:r) => rag))
        @test_throws ArgumentError TreeData(ragchain)
        schain = FlexiChain{Symbol}(10, 2, Dict(Parameter(:s) => fill("x", 10, 2)))
        @test_throws ArgumentError TreeData(schain)
    end

    @testset "ingest empty chain" begin
        chain = FlexiChain{Symbol}(10, 2, Dict{Parameter{Symbol},Matrix{Float64}}())
        X = TreeData(chain)
        @test isempty(propertynames(X))
        @test map(name, TreeArrays.dims(X)) == (:draw, :chain, :param)
    end

    @testset "emit dense leaf round-trip" begin
        arr = reshape(Float64.(1:60), 10, 2, 3)
        X = TreeData(arr, :draw, :chain, :param => [:a, :b, :c])
        c = FlexiChain(X)
        @test parameters(c) == [:a, :b, :c]
        @test (niters(c), nchains(c)) == (10, 2)
        @test parent(c[Parameter(:b)]) == arr[:, :, 2]
        # any axis order emits identically (permuted as needed)
        Xp = TreeData(permutedims(arr, (3, 1, 2)), :param => [:a, :b, :c], :draw, :chain)
        cp = FlexiChain(Xp)
        @test parameters(cp) == [:a, :b, :c]
        @test parent(cp[Parameter(:b)]) == arr[:, :, 2]
        # string labels convert; duplicates and non-integer coords throw
        Xs = TreeData(arr, :draw, :chain, :param => ["a", "b", "c"])
        @test parameters(FlexiChain(Xs)) == [:a, :b, :c]
        Xd = TreeData(arr, :draw, :chain, :param => [:a, :b, "a"])
        @test_throws ArgumentError FlexiChain(Xd)
        Xw = TreeData(arr, :draw => string.(1:10), :chain, :param => [:a, :b, :c])
        @test_throws ArgumentError FlexiChain(Xw)
        # offset coords round-trip; unlabelled axes default to 1:n
        Xo = TreeData(arr, :draw => collect(101:110), :chain, :param => [:a, :b, :c])
        @test collect(iter_indices(FlexiChain(Xo))) == collect(101:110)
        @test collect(chain_indices(FlexiChain(Xo))) == collect(1:2)
        # fixed dims are allowed (documented: they do not survive)
        Xf = TreeData(arr, :draw, :chain, :param => [:a, :b, :c]; dose=20)
        @test parameters(FlexiChain(Xf)) == [:a, :b, :c]
        # wrong shapes throw by name
        X2 = TreeData(arr[:, :, 1:2], :draw, :subject => [:s1, :s2], :chain)
        m2 = TreeData(reshape(arr[:, :, 1], 10, 2), :draw, :chain)
        @test_throws ArgumentError FlexiChain(m2)
        @test_throws ArgumentError FlexiChain(X2)
    end

    @testset "emit record round-trip" begin
        arr = reshape(Float64.(1:40), 10, 2, 2)
        chain = FlexiChain{Symbol}(arr, (Parameter(:x), Parameter(:y)))
        c2 = FlexiChain(TreeData(chain))
        @test parameters(c2) == [:x, :y]
        @test parent(c2[Parameter(:x)]) == arr[:, :, 1]
        @test parent(c2[Parameter(:y)]) == arr[:, :, 2]
        # nested fields unstack back to array-valued keys
        mz = [Float64[100i + j, 200i + j] for i in 1:10, j in 1:2]
        achain = FlexiChain{Symbol}(10, 2, Dict(Parameter(:z) => mz))
        back = FlexiChain(TreeData(achain))
        @test parameters(back) == [:z]
        @test parent(back[Parameter(:z)])[3, 2] == mz[3, 2]
        # numeric fields promote (like FC's own Long format); the rest throws
        mix = TreeData(:param => (; i=fill(7, 10, 2), f=fill(1.5, 10, 2)), :draw, :chain)
        cm = FlexiChain(mix)
        @test eltype(parent(cm[Parameter(:i)])) == Float64
        @test parent(cm[Parameter(:i)])[1, 1] == 7.0
        str = TreeData(:param => (; s=fill("x", 10, 2)), :draw, :chain)
        @test_throws ArgumentError FlexiChain(str)
        sca = TreeData(:param => (; s=1.0), :draw, :chain)
        @test_throws ArgumentError FlexiChain(sca)
        inner = TreeData(:param => (; x=reshape(Float64.(1:20), 10, 2)), :draw, :subject => [1, 2])
        @test_throws ArgumentError FlexiChain(inner)
        anon = TreeData((; x=reshape(Float64.(1:20), 10, 2)), :draw, :chain)
        @test_throws ArgumentError FlexiChain(anon)
    end

    @testset "emit guards" begin
        rag = TreeData([TreeData([1.0, 2.0], :draw), TreeData([1.0], :draw)], :subject => [:s1, :s2])
        @test_throws ArgumentError FlexiChain(rag)
        tup = TreeData((fill(1.0, 10, 2), fill(2.0, 10, 2)), :draw, :chain)
        @test_throws ArgumentError FlexiChain(tup)
    end

    @testset "melt summaries" begin
        arr = reshape(Float64.(1:60), 10, 2, 3)
        chain = FlexiChain{Symbol}(arr, (Parameter(:x), Parameter(:y), Extra(:lp)))
        st = summarystats(chain)
        S = TreeData(st)
        @test name(outerdim(S)) == :param
        @test :stat in map(name, TreeArrays.dims(S))
        @test :mean in coords(S, :stat)
        @test parent(S).x == parent(st[Parameter(:x)])
        @test length(parent(S).x) == length(coords(S, :stat))
        # single-stat full collapse melts to a dense :param vector
        mn = mean(chain)
        m = TreeData(mn)
        @test parent(m) isa AbstractVector
        got = Dict(zip(coords(m, :param), parent(m)))
        for k in keys(mn)
            @test got[Symbol(k)] == mn[k]
        end
        # partial collapse keeps the surviving sample dims
        mc = mean(chain; dims=:chain)
        C = TreeData(mc)
        @test map(name, TreeArrays.dims(C))[1] == :draw
        @test coords(C, :draw) == collect(1:10)
        @test parent(C).y == parent(mc[Parameter(:y)])
        # multi-stat partial collapse rides over (:draw, :stat)
        cs = collapse(chain, [mean, sum]; dims=:chain)
        CS = TreeData(cs)
        @test map(name, TreeArrays.dims(CS))[1:2] == (:draw, :stat)
        @test size(parent(CS).x, 1) == 10
        @test size(parent(CS).x, 2) == 2
    end

    @testset "melt preserves missing stats" begin
        mx = reshape(Float64.(1:20), 10, 2)
        chain = FlexiChain{Symbol}(10, 2,
            Dict(Parameter(:x) => mx, Parameter(:s) => fill("v", 10, 2)))
        cs = collapse(chain, [mean, length]; dims=:both)
        S = TreeData(cs)
        @test length(parent(S).s) == 2
        @test any(ismissing, parent(S).s)
        @test !any(ismissing, parent(S).x)
    end

end
