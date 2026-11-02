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
        :setdim,   # under decision for 1.0: an exported stub that errors on every call
    ])
    @test Set(names(TreeArrays)) == frozen
    # `quantile!` was a stray re-export of `Statistics.quantile!` with no TreeArrays method.
    @test :quantile! ∉ names(TreeArrays)
    # The dimension-less `Base.cat(X::TreeData...)` is gone: no TreeArrays method on `cat`.
    @test !any(m -> m.module === TreeArrays, methods(Base.cat))
    # Internal accessors stay unexported.
    @test :meta ∉ names(TreeArrays)
    @test :name ∉ names(TreeArrays)
end
