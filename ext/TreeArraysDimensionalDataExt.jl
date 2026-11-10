module TreeArraysDimensionalDataExt

# `TreeData(A::AbstractDimArray; axes, fixed dims...)`: a DimensionalData array becomes a tree
# whose outer axes are A's dims, BY NAME, labelled with their lookups (decision 0jwqiy5). Its
# cells are converted once by `TreeArrays._fromcells` (src/convert.jl): numbers stay a zero-copy
# dense leaf, NamedTuple / array cells -- a model's return values, `returned(model, chain)` --
# become records, dense leaves and ragged nodes. FlexiChains names the iteration dim `:iter`;
# it becomes `:draw`, TreeArrays' name for it (as in `TreeData(chain)`), so chains and their
# return values reduce with the same `dims = (:draw, :chain)`.

using TreeArrays
using DimensionalData
const DD = DimensionalData

function TreeArrays.TreeData(A::DD.AbstractDimArray; axes = nothing, kw...)
    X = TreeArrays._fromcells(parent(A), map(_treedim, DD.dims(A)), axes, ())
    isempty(kw) ? X : TreeData(X, map(((k, v),) -> TreeDim(k, v), collect(pairs(kw)))...)
end

_treedim(d) = TreeDim(_treename(DD.name(d)), _treecoords(DD.lookup(d)))
_treename(n::Symbol) = n === :iter ? :draw : n
_treecoords(::DD.Lookups.NoLookup) = missing      # no labels: an unlabelled axis, not 1:n
_treecoords(l) = parent(l)

end
