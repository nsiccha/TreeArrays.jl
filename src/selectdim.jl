# ===================== selectdim: restrict a named axis to a subset =====================
# The SELECTION dual of the `dims=` reductions (mapslices/mean/sum/quantile): those COLLAPSE a
# named axis; this RESTRICTS one to the coordinates a selector keeps, leaving every other axis
# intact. It is the TreeArrays spelling of Bruno draws_core's `subdf(pattern::Regex)`
# (`df_names[map(contains(pattern), df_names)]`, draws_core.jl:322) -- an axis-filter over a
# ~600k-label `:param` axis, the most-used non-reduction op in the DataFrames->TA migration
# (snag `named-axis-label`).
#
# NO DATA COPY (the snag's hard requirement): the backing array is sliced with a `view`, so a
# subset of a `(draw x chain x 600k)` matrix never materializes -- only the axis's own (tiny)
# coordinate labels are subset. This is why it lives on `TreeArray` and threads the physical
# axis position, rather than delegating to `getindex` (which drops every dim, array_interface.jl).
#
# Selector polymorphism mirrors `subdf_pattern`'s Regex-vs-mask dispatch, and adds predicates
# and integer indices:
#   Regex                     -> keep coords `contains(pattern)` matches      (LABELLED axis only)
#   `label -> Bool` predicate -> keep coords the predicate accepts            (LABELLED axis only)
#   AbstractVector{Bool}      -> an explicit mask (length must match the axis)
#   AbstractVector{<:Integer} -> explicit positions (may reorder / repeat)
# The last two are positional, so they also restrict an UNLABELLED axis (the reporter's `:draw`
# and `:chain` are unlabelled) -- the axis stays unlabelled, its length just shrinks.
#
# `TreeArray` also covers a ragged tree's OUTER axis (its parent IS an array of sub-trees), so
# restricting e.g. a `:subject` axis by label works and stays ragged; an INNER axis (inside the
# leaves) or a record axis is NOT an outer array axis and is refused BY NAME below rather than
# silently ignored -- the same "never silently return a valid-looking value" rule the reductions
# hold (decision 16fwcnx).

_checkmask(sel, n) = length(sel) == n ? sel :
    error("TreeArrays: selectdim Bool mask has length $(length(sel)) but the axis has $n coordinates.")

# --- Resolve a selector against a LABELLED axis's coords (always an `AbstractVector` here -- a
# Tuple axis is collected first, so every returned index is a `Vector` and `view`s cleanly).
_selectindex(sel::Regex, coords::AbstractVector)                = map(contains(sel), coords)
_selectindex(sel::AbstractVector{Bool}, coords::AbstractVector) = _checkmask(sel, length(coords))
_selectindex(sel::AbstractVector{<:Integer}, coords::AbstractVector) = sel
_selectindex(sel::AbstractString, ::AbstractVector) = error(
    "TreeArrays: selectdim: a bare String selector `\"$sel\"` is ambiguous (exact vs substring). " *
    "Use a Regex (`r\"$sel\"`), a predicate (`==(\"$sel\")` or `contains(\"$sel\")`), a Bool mask, " *
    "or integer indices.")
function _selectindex(sel, coords::AbstractVector)   # a predicate: label -> Bool
    mask = map(sel, coords)
    eltype(mask) <: Bool || error(
        "TreeArrays: selectdim: a predicate over a label axis must return Bool; `$(sel)` returned " *
        "eltype $(eltype(mask)). Pass a Regex, a `label -> Bool` predicate, a Bool mask, or integer indices.")
    mask
end

# --- Resolve a selector against an UNLABELLED axis (`values === missing`): only positional
# Bool/Int selectors make sense -- there are no labels to match. `axlen` comes from the array.
_positionalindex(sel::AbstractVector{Bool}, axlen::Int)      = _checkmask(sel, axlen)
_positionalindex(sel::AbstractVector{<:Integer}, ::Int)      = sel
_positionalindex(sel, ::Int) = error(
    "TreeArrays: selectdim cannot filter an unlabelled axis (values === missing) by `$(typeof(sel))` -- " *
    "it has no coordinates to match. Give the axis labels at construction (`:name => labels`) to select " *
    "by Regex/predicate, or pass a positional Bool mask / integer indices to subset it by position.")

# keep-as-provided (mirrors quantile's level handling, treearrays-use §2): a Tuple axis stays a
# Tuple, a Vector stays a Vector. A filtered AbstractRange cannot stay a range, so it becomes a
# Vector -- inherent, and ranges carry no such invariant.
_subsetcoords(vals::Tuple, coordvec, idx) = Tuple(coordvec[idx])
_subsetcoords(vals, coordvec, idx)        = coordvec[idx]

# `Pair` form (matches TreeArrays' pervasive `name => value` idiom: `TreeData(x, :p => coords)`,
# `quantile(X, :band => (...))`) and a 3-arg form mirroring `Base.selectdim(A, d, i)` with a NAME
# in place of the dimension number. Extends `Base.selectdim` (a method addition -- no export, no
# name clash) because the spirit is identical: select along a dimension, return a view.
"""
    selectdim(X::TreeData, name => selector)
    selectdim(X::TreeData, name::Symbol, selector)

Restrict a named axis to the coordinates `selector` keeps, leaving every other
axis intact — the **selection dual** of the `dims =` reductions, which collapse an
axis instead.

**No data is copied.** The backing array is sliced with a `view`, so a subset of a
`(draw × chain × 600_000)` matrix never materializes; only the axis's own (tiny)
coordinate labels are subset.

```julia
selectdim(X, :param => r"^unit_params_")      # Regex: keep labels it `contains`-matches
selectdim(X, :param => startswith("beta"))    # a `label -> Bool` predicate
selectdim(X, :param => boolmask)              # an explicit Bool mask (length == axis)
selectdim(X, :param => [8, 1, 2])             # integer indices — subset and/or reorder
selectdim(X, :param, sel)                     # 3-arg form, mirroring Base.selectdim(A, d, i)
```

The filtered axis carries exactly the matched labels, keep-as-provided (a `Tuple`
stays a `Tuple`; a filtered range necessarily becomes a `Vector`). The result is
a [`TreeData`](@ref), so it composes with the reductions:
`mean(selectdim(X, :param => r"…"); dims = :draw)`.

**Positional** selectors (a Bool mask, integer indices) also restrict an
*unlabelled* axis — it stays unlabelled and merely gets shorter. A Regex or
predicate on one errors: there are no labels to match.

Like `dims =`, this is *foundALL* and never silently wrong. Each of these throws
by name: a typo'd axis name; a bare `String` selector (ambiguous between exact
and substring — use `==("x")`, `contains("x")` or a Regex); a predicate that does
not return `Bool`; a wrong-length mask; and selecting an inner or record axis
(only the outer array axes are selectable, a ragged tree's outer axis included).
"""
Base.selectdim(X::TreeArray, (nm, sel)::Pair{Symbol}) = _selectdim(X, nm, sel)
Base.selectdim(X::TreeArray, nm::Symbol, sel)         = _selectdim(X, nm, sel)

function _selectdim(X::TreeArray, nm::Symbol, sel)
    alldims = TreeArrays.dims(X)
    n_ax    = ndims(parent(X))
    # Only the first `n_ax` dims are real array axes (the `_reduceouter` invariant, mapslices.jl);
    # fixed/ghost dims trail after and have no axis to slice.
    pos     = findfirst(i -> name(alldims[i]) === nm && _isaxis(alldims[i]), 1:n_ax)
    isnothing(pos) && _noselectaxis(X, nm)
    vals = meta(alldims[pos]).values
    if vals isa Union{Tuple,AbstractArray,AbstractRange}          # labelled axis
        coordvec = vals isa Tuple ? collect(vals) : vals
        idx      = _selectindex(sel, coordvec)
        newdim   = TreeDim(nm, _subsetcoords(vals, coordvec, idx))
    else                                                          # unlabelled axis (values === missing)
        idx      = _positionalindex(sel, size(parent(X), pos))
        newdim   = TreeDim(nm)                                    # stays unlabelled, just shorter
    end
    newparent = view(parent(X), ntuple(i -> i == pos ? idx : Colon(), n_ax)...)   # NO data copy
    newdims   = ntuple(i -> i == pos ? newdim : alldims[i], length(alldims))
    TreeData(newparent, merge(meta(X), (;dims = newdims)))
end

@noinline function _noselectaxis(X::TreeArray, nm::Symbol)
    n_ax  = ndims(parent(X))
    outer = ntuple(i -> name(TreeArrays.dims(X)[i]), n_ax)
    allnames, _ = _alldimnames(typeof(X))
    nm in allnames && error(
        "TreeArrays: selectdim cannot restrict `:$nm` -- it is not one of this tree's selectable " *
        "outer array axes $(outer) (it is a fixed/aggregated dim, or lives inside the leaves/record). " *
        "Selecting an inner or record axis is not implemented yet -- ask TreeArrays.")
    error(
        "TreeArrays: selectdim: no axis named `:$nm`. Selectable (outer array) axes: $(outer). " *
        "`selectdim` is foundALL -- a name that resolves nowhere is a typo.")
end
