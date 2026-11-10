# Changelog

All notable changes to TreeArrays.jl are documented here, newest first.

Every landing that adds or changes public API (an export, a macro, a keyword, a
method, or documented behaviour a dependant can rely on) bumps the patch digit
in `Project.toml` and adds its entry under a new heading at the top of this
file. An entry that breaks something a dependant relies on says **Breaking:**
and what to change. Before 1.0 the minor digit is the breaking one in
[semantic versioning](https://semver.org/). A package or app that calls that
API states the version as its floor, for example `[compat] TreeArrays =
"0.1.1"`. An environment that pins an older TreeArrays commit is then refused
when it resolves, instead of failing or silently behaving differently at run
time.

## 0.2.3

- New: `TreeData(A::AbstractDimArray; axes = nothing, fixed_dims...)` through a
  DimensionalData weak-dependency extension. A's dims become named axes with
  their lookups as labels; the iteration dim `:iter` becomes `:draw`. Number
  cells stay a zero-copy leaf. One NamedTuple or array per cell, such as a model's
  return values from `returned(model, chain)`, is converted once into records,
  dense leaves and ragged nodes (arrays of arrays, e.g. participants with
  different numbers of trials). `axes` names the inner axes, mirroring the
  value's structure. A plain array of NamedTuples is still refused; the message
  now points to this route.
- New: `size(X, name)` and `axes(X, name)` read an axis length by dim name, on
  every tree shape.
- Changed: broadcasting over a tree (`exp.(X)`, `X .- 1`,
  `X .- mean(X; dims = :draw)`) now returns a tree with the same dims instead of
  a plain `Array` that dropped every label. Trees combine when their axes match
  by name, order and labels; a reduced or fixed dim stretches along the other
  tree's axis; records go field by field, ragged trees leaf by leaf. A plain
  array combined with a tree, and mismatched axes, refuse by name. Code that
  needs the plain array writes `parent(X) .- 1`. The old result was undocumented
  (a side effect of the forwarded array interface), and a scan of every known
  dependant found none relying on it.

## 0.2.2

- New keyword `align = nothing | :union | :intersect` on `mapslices` and every
  built-in reducer (`mean`, `sum`, `var`, `std`, `quantile`, `hdi`,
  `nanquantile`). It says how a reduction over a ragged tree's outer axis handles
  a **kept** axis whose labels differ between the leaves, for example
  participants with different numbers of trials. `:union` pools, at every label
  any leaf carries, the leaves that carry it; `:intersect` keeps only the labels
  every leaf carries. Labels are matched by value, never by position. The
  default still refuses by name, and the message now names `align`.
- Fix: reducing a ragged tree's outer axis **alone** (`dims = :subject`) checked
  only that the leaves had the same size. Leaves with the same size but
  different labels (subject 1 sampled at `t = 1.0`, subject 2 at `t = 1.5`) were
  pooled position by position under the first leaf's labels, with no error. That
  now refuses by name, unless `align` is given. The pooled reduction
  (`dims = (:draw, :subject)`) already checked labels.
- Fix: reducing a ragged tree's outer axis whose leaves are records with array
  fields gathered the whole field arrays instead of their elements. `mean`, `sum`
  and `std` happened to work on whole arrays; `quantile`, `hdi` and custom
  kernels such as `maximum` threw. Such fields are now reduced position by
  position.

## 0.2.1

- `TreeActualArray` now exposes numeric scalar-backed leaves, including fully
  reduced results, as zero-dimensional, read-only array views. `A[]` reads the
  stored scalar without copying; `parent(A)` retains the tree and `dims(A)`
  preserves fixed and reduced dimensions. `collect(A)` and `Array(A)` produce
  independent zero-dimensional arrays. Scalar leaves claiming real axes fail
  descriptively instead of silently dropping those axes.

## 0.2.0

Three corners of the surface, listed as "being decided" in 0.1.2, are settled
(user decision: the TreeArrays agent's recommendations, 2026-11-02). All three
are **Breaking** for a caller that used the old behaviour; no dependant did.

- **Breaking:** `setdim` is removed. It was an exported stub that errored on
  every call since `bfc1de6`. A lazy re-label / fix-a-dim operation can be
  added later without a breaking change; `selectdim` restricts an axis today.
- **Breaking:** `TreeData(:time)` with dim names and no backing array now
  throws by name. It used to return a builder closure meant for
  `map(TreeData(:time), values, times)`, a constructor call that silently gave
  back a non-tree. Write the closure out:
  `map((v, t) -> TreeData(v, :time => t), values, times)`.
- **Breaking:** an `AbstractArray{<:NamedTuple}` backing (an array of structs),
  or a `Tuple` of `NamedTuple`s, is refused by name at construction. It matched none of the four leaf shapes
  and failed late inside a reduction (`float(::Type{Any})`). Records are stored
  struct-of-arrays: build `TreeData(:field => (; a = …, b = …), dims...)`, or
  wrap each element in its own record leaf,
  `TreeData(map(nt -> TreeData(:field => nt), x), dims...)`.
- **Breaking:** the same refusal applies where TreeArrays itself would have
  built that shape: a `mapslices` kernel, or a `map` / `stack` sweep cell, that
  returns a plain `NamedTuple` now throws by name instead of gathering into an
  array of structs that the next reduction could not read. Return
  `TreeData(:stat => nt)` (a named record leaf, as `quantile(X, p::NamedTuple)`
  and `hdi` do) or a scalar. `@kernel` already wraps its record and is unchanged.
- `src/setdim.jl` is renamed `src/sweeps.jl`; it holds only the `map` /
  `stack` scenario sweeps, which are otherwise unchanged.

## 0.1.2

- The public API is now listed on the documentation's API page under "Public
  API (frozen for 1.0)", and a test (`test/public_api.jl`) holds the export
  list to exactly that list. Anything not listed there is internal and may
  change without notice; that includes `TreeArrays.meta`, `TreeArrays.name`
  and the layout of a tree's `meta` NamedTuple.
- `quantile!` is no longer exported. TreeArrays never defined a method for it;
  the export only re-exported `Statistics.quantile!`, and no dependant used it
  through TreeArrays. A caller that did adds `using Statistics: quantile!`.
- The undocumented `Base.cat(X::TreeData...)` method is removed. It wrapped its
  arguments in a dimension-less `TreeTuple`; nothing used it. Build a positional
  container with `TreeData((X, Y), :name => (…, …))`, or a sweep with
  `map(f, TreeDim(...))`.
- Three corners of the surface are being decided before 1.0 and are marked as
  such on the API page: the exported `setdim` stub, the `TreeData(:time)`
  currying builder, and whether an array-of-structs backing is refused by name.

## 0.1.1

- A reduction with a scalar probability, such as
  `quantile(X, TreeDim(:median, 0.5); dims=:draw)`, can now be reduced again
  with `nanquantile`. Each position of that result holds a scalar, and the
  chained `nanquantile(result, TreeDim(:median2, 0.5); dims=:param)` used to
  throw `MethodError: no method matching float(::Type{Any})`; it now returns
  the median of medians. A chained `quantile` already worked, and now also
  sorts in a buffer of the data's own element type instead of an `Any` buffer
  (`c4cfcbb`).

## 0.1.0

The version every commit up to `3dfeb9f` carried. Its changes are not listed
here; the [documentation](https://nsiccha.github.io/TreeArrays.jl/dev/) describes
what it does.
