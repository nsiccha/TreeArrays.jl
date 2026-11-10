# `align=` (decision 1ibupji): pool a ragged outer axis while KEEPING an inner axis whose labels
# differ across leaves -- by label, never by position. Every value is checked against the same
# reduction over a hand-pooled plain-Julia bag.
@testitem "align= across ragged leaves" begin
    using Statistics, Tables, NaNStatistics
    ND, NC = 50, 2
    Tp = [6, 6, 4, 5]; NP = length(Tp)
    pe = [randn(ND, NC, Tp[p]) for p in 1:NP]
    rpe = TreeData([TreeData(pe[p], :draw, :chain, :time => 1:Tp[p]) for p in 1:NP], :participant => 1:NP)
    bands = (lower = 0.025, median = 0.5, upper = 0.975)
    pool(t; ps = findall(>=(t), Tp)) = reduce(vcat, [vec(pe[p][:, :, t]) for p in ps])
    @testset "default still refuses, names align" begin
        @test_throws "align = :union" quantile(rpe, bands; dims = (:draw, :chain, :participant))
        @test_throws "align = :union" mean(rpe; dims = :participant)
        @test_throws "not an alignment policy" mean(rpe; dims = :participant, align = :outer)
        @test_throws "not an alignment policy" mean(rpe; dims = :participant, align = "union")
    end
    @testset "straddle union / intersect, byte-exact" begin
        q = quantile(rpe, bands; dims = (:draw, :chain, :participant), align = :union)
        @test coords(q, :time) == 1:6
        c = Tables.columns(TreeTable(q))
        for t in 1:6
            @test [c.lower[t], c.median[t], c.upper[t]] == quantile(pool(t), [0.025, 0.5, 0.975])
        end
        qi = quantile(rpe, bands; dims = (:draw, :chain, :participant), align = :intersect)
        @test coords(qi, :time) == 1:4
        ci = Tables.columns(TreeTable(qi))
        @test [ci.median[t] for t in 1:4] == [quantile(pool(t), 0.5) for t in 1:4]
        m = mean(rpe; dims = (:draw, :chain, :participant), align = :union)
        @test parent(m) ≈ [mean(pool(t)) for t in 1:6]
        @test parent.(parent(nanquantile(rpe, TreeDim(:q, 0.5); dims = (:draw, :chain, :participant), align = :union))) == [quantile(pool(t), 0.5) for t in 1:6]
        @test parent(mean(rpe; dims = (:draw, :chain, :participant), align = :union, threads = false)) == parent(m)
    end
    @testset "outer-only gather union" begin
        m = mean(rpe; dims = :participant, align = :union)
        @test coords(m, :time) == 1:6
        P = parent(m)
        @test size(P) == (ND, NC, 6)
        @test all(P[d, ch, t] ≈ mean([pe[p][d, ch, t] for p in findall(>=(t), Tp)]) for d in 1:ND, ch in 1:NC, t in 1:6)
        mi = mean(rpe; dims = :participant, align = :intersect)
        @test size(parent(mi)) == (ND, NC, 4)
        # chained: participant-mean trajectory, then its posterior band
        pm = quantile(m, bands; dims = (:draw, :chain))
        @test coords(pm, :time) == 1:6
    end
    @testset "record leaves (Peter's shape)" begin
        ex = [randn(ND, NC, Tp[p]) for p in 1:NP]
        rs = TreeData([TreeData(:state => (; expectation = ex[p], prediction_error = pe[p]), :draw, :chain, :time => 1:Tp[p]) for p in 1:NP], :participant => 1:NP)
        m = mean(rs; dims = :participant, align = :union)
        @test coords(m.prediction_error, :time) == 1:6
        @test parent(m.prediction_error) ≈ parent(mean(rpe; dims = :participant, align = :union))
        @test coords(m, :time) == 1:6       # the record's own dims carry the aligned axis
        # conformable record leaves: quantile used to crash (convert Array -> Float64)
        eq = TreeData([TreeData(:state => (; a = randn(ND, NC, 3), b = randn(ND, NC, 3)), :draw, :chain, :time => 1:3) for p in 1:NP], :participant => 1:NP)
        qm = quantile(eq, TreeDim(:q, 0.5); dims = :participant)
        A = [parent(parent(eq)[p]).a for p in 1:NP]
        @test parent(parent(qm.a)[2, 1, 3]) == quantile([A[p][2, 1, 3] for p in 1:NP], 0.5)
        @test parent(mean(eq; dims = :participant).a) ≈ sum(A) ./ NP
        # straddle over record leaves still refuses, now pointing at X.field
        @test_throws "X.field" quantile(rs, bands; dims = (:draw, :chain, :participant), align = :union)
        q = quantile(rs.prediction_error, bands; dims = (:draw, :chain, :participant), align = :union)
        @test coords(q, :time) == 1:6
    end
    @testset "labels, not positions" begin
        s1 = TreeData([1.0, 10.0, 100.0], :time => [0.5, 1.0, 2.0])
        s2 = TreeData([2.0, 20.0, 200.0], :time => [0.5, 1.5, 4.0])
        X = TreeData([s1, s2], :subject => 1:2)
        @test_throws "labels its `:time` axis differently" mean(X; dims = :subject)   # was a silent mislabel
        u = mean(X; dims = :subject, align = :union)
        @test coords(u, :time) == [0.5, 1.0, 1.5, 2.0, 4.0]
        @test parent(u) == [1.5, 10.0, 20.0, 100.0, 200.0]
        @test parent(mean(X; dims = :subject, align = :intersect)) == [1.5]
        # coords=true sees only the contributing subjects' labels
        w = mapslices((y, s) -> sum(s), X; dims = :subject, coords = true, align = :union)
        @test parent(w) == [3, 1, 2, 1, 2]
        # non-numeric labels keep first-met order
        a = TreeData([1.0, 2.0], :k => [:b, :a]); b = TreeData([3.0, 4.0], :k => [:c, :a])
        @test coords(mean(TreeData([a, b], :g => 1:2); dims = :g, align = :union), :k) == [:b, :a, :c]
    end
    @testset "refusals" begin
        U = TreeData([TreeData(randn(3), :time), TreeData(randn(4), :time)], :s => 1:2)
        @test_throws "unlabelled" mean(U; dims = :s, align = :union)
        D = TreeData([TreeData(randn(3), :time => [1, 1, 2]), TreeData(randn(2), :time => [1, 2])], :s => 1:2)
        @test_throws "repeats" mean(D; dims = :s, align = :union)
        E = TreeData([TreeData(randn(2), :time => [1, 2]), TreeData(randn(2), :time => [3, 4])], :s => 1:2)
        @test_throws "share no" mean(E; dims = :s, align = :intersect)
        @test parent(mean(E; dims = :s, align = :union)) == vcat(parent(parent(E)[1]), parent(parent(E)[2]))
    end
    @testset "conformable input: align is a no-op" begin
        R = TreeData([TreeData(randn(ND, NC, 5), :draw, :chain, :time => 1:5) for p in 1:3], :participant => 1:3)
        @test parent(quantile(R, bands; dims = (:draw, :chain, :participant), align = :union).median) ==
                    parent(quantile(R, bands; dims = (:draw, :chain, :participant)).median)
        @test parent(mean(R; dims = :participant, align = :intersect)) == parent(mean(R; dims = :participant))
    end
    @testset "melt of an aligned result" begin
        pm = mean(rpe; dims = (:draw, :chain), align = :union)   # inner-only: align unused, ragged melt
        @test length(Tables.columns(pm).value) == sum(Tp)
        q = quantile(rpe, bands; dims = (:draw, :chain, :participant), align = :union)
        @test Tables.columnnames(Tables.columns(TreeTable(q))) == (:time, :lower, :median, :upper)
    end
end
