# Decision 0jwqiy5 (usability): `size(X, :draw)` / `axes(X, :draw)` on every tree shape.
@testitem "size and axes by dim name" begin
    using Statistics
    dn(X) = map(d -> typeof(d).parameters[1], TreeArrays.dims(X))
    X = TreeData(randn(100, 4, 3), :draw, :chain, :participant => 1:3)
    @test size(X, :draw) == 100 && size(X, :participant) == 3 && axes(X, :chain) == Base.OneTo(4)
    @test size(X, 2) == 4   # positional unchanged
    P = TreeData(:param => (; lr = rand(100, 4, 3), noise = rand(100, 4, 3)), :draw, :chain, :participant => 1:3)
    @test size(P, :participant) == 3 && size(P, :param) == 2
    rpe = TreeData([TreeData(randn(50, 2, n), :draw, :chain, :time => 1:n) for n in (6, 6, 4)], :participant => 1:3)
    @test size(rpe, :participant) == 3 && size(rpe, :draw) == 50
    @test_throws "differs in length" size(rpe, :time)
    @test_throws "no dim `:drw`" size(X, :drw)
    @test_throws "reduced dim" size(mean(X; dims = :draw), :draw)
    rs = TreeData([TreeData(:state => (; a = randn(5, 2, n), b = randn(5, 2, n)), :draw, :chain, :time => 1:n) for n in (3, 3)], :p => 1:2)
    @test size(rs, :time) == 3 && size(rs, :state) == 2
end

# Decision 0jwqiy5: broadcasting over trees returns a tree with the same dims; reduced dims stretch;
# mismatched axes and plain arrays refuse by name. Values checked against the same broadcast on the backing.
@testitem "broadcasting keeps the labels" begin
    using Statistics
    dn(X) = map(d -> typeof(d).parameters[1], TreeArrays.dims(X))
    A = randn(100, 4, 3)
    X = TreeData(A, :draw, :chain, :participant => 1:3)
    @testset "broadcast" begin
        E = exp.(X); @test E isa TreeData && parent(E) == exp.(A) && dn(E) == dn(X)
        @test parent(X .- 1) == A .- 1
        @test parent(exp.(X .- 1) .* 2) == exp.(A .- 1) .* 2           # fused
        C = X .- mean(X; dims = :draw)                                 # centring via a reduced dim
        @test C isa TreeData && dn(C) == dn(X) && parent(C) ≈ A .- mean(A; dims = 1)
        S = X ./ std(X; dims = (:draw, :chain))
        @test parent(S) ≈ A ./ std(A; dims = (1, 2))
        @test parent(X .+ X) == 2A
        @test parent(X .> 0) == (A .> 0)
        @test parent(X .+ Ref(2.0)) == A .+ 2
        P = TreeData(:param => (; lr = rand(10, 2), noise = rand(10, 2)), :draw, :chain)
        Q = log.(P); @test Q isa TreeNamedTuple && parent(Q).lr == log.(parent(P).lr)
        @test parent(P .- mean(P; dims = :draw)).noise ≈ parent(P).noise .- mean(parent(P).noise; dims = 1)
        @test (P.lr .* 2) isa TreeData
        rpe = TreeData([TreeData(randn(5, 2, n), :draw, :chain, :time => 1:n) for n in (3, 4)], :p => 1:2)
        R = rpe .- mean(rpe; dims = :draw)
        @test R isa TreeRaggedArray && size(parent(parent(R)[2])) == (5, 2, 4)
        @test parent(parent(R)[2]) ≈ parent(parent(rpe)[2]) .- mean(parent(parent(rpe)[2]); dims = 1)
        # refusals
        @test_throws "plain Array" X .+ randn(100, 4)
        Y = TreeData(randn(4, 100, 3), :chain, :draw, :participant => 1:3)
        @test_throws "different order" X .+ Y
        W = TreeData(randn(3), :participant => 1:3)
        @test_throws "absent from" X .* W
        Z = TreeData(randn(100, 4, 3), :draw, :chain, :participant => 4:6)
        @test_throws "differs between" X .+ Z
        @test_throws "outer axes" rpe .+ TreeData([TreeData(randn(5, 2, n), :draw, :chain, :time => 1:n) for n in (3, 4)], :q => 1:2)
    end
end

# Decision 0jwqiy5: `TreeData(::AbstractDimArray; axes)` -- dims by name, `:iter` -> `:draw`, one NamedTuple
# per draw (a `returned(model, chain)` DimMatrix) transposed into records, dense and ragged leaves.
@testitem "DimensionalData arrays and returned values convert once" begin
    using Statistics, DimensionalData, Tables
    dn(X) = map(d -> typeof(d).parameters[1], TreeArrays.dims(X))
    @testset "DimArray -> tree" begin
        D = DimArray(randn(100, 4, 3), (Dim{:draw}(1:100), Dim{:chain}(1:4), Dim{:participant}([:a, :b, :c])))
        X = TreeData(D)
        @test dn(X) == (:draw, :chain, :participant) && parent(X) === parent(D) && coords(X, :participant) == [:a, :b, :c]
        @test dn(TreeData(DimArray(rand(3), DimensionalData.X()))) == (:X,) && TreeArrays.dims(TreeData(DimArray(rand(3), DimensionalData.X())))[1] == TreeDim(:X)
        @test dn(TreeData(D; dose = 20)) == (:draw, :chain, :participant, :dose)
        # a returned(model, chain)-shaped DimMatrix: (iter x chain), one NamedTuple per draw
        Tp = [5, 5, 3]
        vals = [(; mu = randn(), pe = randn(3, 5), traj = [randn(t) for t in Tp]) for i in 1:100, c in 1:4]
        rets = DimArray(vals, (Dim{:iter}(1:100), Dim{:chain}(1:4)))
        R = TreeData(rets; axes = (pe = (:participant, :time), traj = (:participant, :trial)))
        @test R isa TreeNamedTuple && keys(parent(R)) == (:mu, :pe, :traj)
        @test parent(R).mu isa Matrix{Float64} && parent(R).mu[7, 2] == vals[7, 2].mu        # raw over (draw, chain)
        @test dn(R.mu) == (:draw, :chain) && coords(R.mu, :draw) == 1:100
        @test dn(R.pe) == (:draw, :chain, :participant, :time) && parent(R.pe)[7, 2, 3, 4] == vals[7, 2].pe[3, 4]
        tr = R.traj
        @test tr isa TreeRaggedArray && dn(tr) == (:participant,) && length(parent(tr)) == 3
        @test dn(parent(tr)[3]) == (:draw, :chain, :trial) && size(parent(parent(tr)[3])) == (100, 4, 3)
        @test parent(parent(tr)[3])[7, 2, 2] == vals[7, 2].traj[3][2]
        m = mean(tr; dims = :participant, align = :union)
        @test coords(m, :trial) == [1, 2, 3, 4, 5] && parent(m)[7, 2, 5] ≈ mean(vals[7, 2].traj[p][5] for p in 1:2)
        q = quantile(tr, (lower = 0.025, upper = 0.975); dims = (:draw, :chain, :participant), align = :union)
        @test coords(q, :trial) == 1:5
        @test parent(mean(R.pe; dims = (:draw, :chain)))[3, 4] ≈ mean(v.pe[3, 4] for v in vals)
        # default names
        R0 = TreeData(rets)
        @test dn(R0.pe) == (:draw, :chain, :elem_1, :elem_2) && dn(R0.traj) == (:elem_1,) && dn(parent(R0.traj)[1]) == (:draw, :chain, :elem_2)
        @test dn(TreeData(DimArray([randn(4) for i in 1:10, c in 1:2], (Dim{:iter}(1:10), Dim{:chain}(1:2))))) == (:draw, :chain, :elem)
        # refusals
        @test_throws "does not have" TreeData(rets; axes = (pee = (:participant, :time),))
        @test_throws "has 2 inner" TreeData(rets; axes = (pe = (:participant,),))
        @test_throws "Tuple of Symbols" TreeData(rets; axes = (pe = (a = 1,),))
        bad = DimArray([(; x = randn(i == 1 ? 2 : 3)) for i in 1:4, c in 1:2], (Dim{:iter}(1:4), Dim{:chain}(1:2)))
        @test_throws "differs between draws" TreeData(bad)
        @test_throws "expected numbers" TreeData(DimArray([(; s = "a") for i in 1:2, c in 1:2], (Dim{:iter}(1:2), Dim{:chain}(1:2))))
        @test_throws "DimArray" TreeData(vals, TreeDim(:draw), TreeDim(:chain))   # plain array of structs: still refused, now pointing here
        # melts / broadcasting compose
        @test length(Tables.columns(mean(tr; dims = (:draw, :chain))).value) == sum(Tp)
        @test parent(exp.(R.pe)) == exp.(parent(R.pe))
    end
end
