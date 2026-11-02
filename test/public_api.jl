# The public surface is what docs/src/api.md lists under "Public API (frozen for 1.0)":
# the export list below plus the Base / Statistics / Tables methods documented there.
# A name added to or dropped from `export` changes what every dependant sees, so it
# changes this list in the same landing, with a CHANGELOG entry (test/changelog.jl).
@testitem "public API surface: the export list is exactly the documented one" begin
    frozen = Set([
        :TreeArrays,
        :TreeDim, :TreeData, :TreeArray, :TreeNamedTuple, :TreeRaggedArray, :TreeTuple,
        :TreeActualArray, :TreeTable,
        :dims, :outerdim, :coords,
        :mapslices, :mean, :sum, :std, :var, :quantile, :hdi,
        Symbol("@kernel"),
    ])
    @test Set(names(TreeArrays)) == frozen
    # `quantile!` was a stray re-export of `Statistics.quantile!` with no TreeArrays method.
    @test :quantile! ∉ names(TreeArrays)
    # The dimension-less `Base.cat(X::TreeData...)` is gone: no TreeArrays method on `cat`.
    @test !any(m -> m.module === TreeArrays, methods(Base.cat))
    # Internal accessors stay unexported.
    @test :meta ∉ names(TreeArrays)
    @test :name ∉ names(TreeArrays)
    # `setdim` was an exported stub that errored on every call; removed in 0.2.0.
    @test :setdim ∉ names(TreeArrays)
    @test !isdefined(TreeArrays, :setdim)
end

# The three corners decision 1c2lsko settled for 1.0 (0.2.0). Each refusal names what it
# refuses and shows the supported spelling; none returns a valid-looking non-tree.
@testitem "1.0 corners: no builder closure, no array-of-structs leaf" begin
    # dim names without a backing used to curry into a builder closure.
    e = try; TreeData(:time); nothing; catch err; err; end
    @test e isa ErrorException
    @test occursin("names dims but no backing array", e.msg)
    @test occursin("map((v, t) -> TreeData(v, :time => t), values, times)", e.msg)
    @test_throws ErrorException TreeData(:draw, :param)
    # the explicit closure is the supported spelling, and builds a ragged tree
    times = [[0.5, 1.0, 2.0], [0.5, 1.5]]
    vals = [randn(3), randn(2)]
    rag = TreeData(map((v, t) -> TreeData(v, :time => t), vals, times), :subject)
    @test rag isa TreeRaggedArray
    @test coords(parent(rag)[2], :time) == [0.5, 1.5]
    # an array of structs is refused by name at construction, through every public form
    recs = [(; a = 1.0, b = 2.0), (; a = 3.0, b = 4.0)]
    e = try; TreeData(recs, :draw); nothing; catch err; err; end
    @test e isa ErrorException
    @test occursin("array or tuple of structs", e.msg)
    @test occursin("TreeData(:field => (; a = …, b = …), dims...)", e.msg)
    @test_throws ErrorException TreeData(recs, TreeDim(:draw))
    @test_throws ErrorException TreeData(recs)
    @test_throws ErrorException TreeData(reshape(recs, 1, 2), :chain, :draw)
    @test_throws ErrorException TreeData(((; a = 1.0), (; a = 2.0)), :draw)   # the positional (Tuple) form
    # the spellings the message recommends work: struct-of-arrays record, or a record leaf per element
    soa = TreeData(:field => (; a = [1.0, 3.0], b = [2.0, 4.0]), :draw)
    @test soa isa TreeNamedTuple && soa.a isa TreeData
    perel = TreeData(map(nt -> TreeData(:field => nt), recs), :draw)
    @test perel isa TreeRaggedArray
    @test length(perel) == 2
    # the same shape is refused where TreeArrays itself would have built it: a kernel or a
    # sweep cell returning a plain NamedTuple (it used to gather into an array of structs and
    # fail on the next reduction with `convert(NamedTuple -> Float64)`).
    X = TreeData(randn(20, 3), :time, :subject)
    e = try; mapslices(y -> (; lo = minimum(y), hi = maximum(y)), X; dims = :time); nothing; catch err; err; end
    @test e isa ErrorException
    @test occursin("plain NamedTuple per slice", e.msg)
    @test occursin("TreeData(:stat => (; lo = …, hi = …))", e.msg)
    @test_throws ErrorException mapslices(y -> (; m = mean(y)), X; dims = (:time, :subject))
    @test_throws ErrorException map(d -> (; a = d), TreeDim(:dose, (20, 200)))
    # the recommended spelling works, and reduces again
    r = mapslices(y -> TreeData(:stat => (; lo = minimum(y), hi = maximum(y))), X; dims = :time)
    @test r isa TreeData && r.lo isa TreeData
    @test mean(r.hi; dims = :subject) isa TreeData
    s = map(d -> TreeData(:stat => (; a = d, b = 2d)), TreeDim(:dose, (20, 200)))
    @test s isa TreeData && quantile(s, (lo = 0.1, hi = 0.9); dims = :dose) isa TreeData
end
