```@meta
CurrentModule = TreeArrays
```

# [Tables & plotting](@id tables)

A `TreeData` **is** a lazy `Tables.jl` source. You can feed one to a plotting
layer, a DataFrame, a CSV writer or anything else that speaks Tables.jl — with no
DataFrames dependency and no hand-rolled melt.

`Tables.istable` holds for *every* `TreeData`. A reduction is not a
precondition, merely the common case; raw draws melt to exactly the per-draw long
rows a spaghetti plot wants. Rectangularity is not a precondition either — a
[ragged tree](@ref "Ragged data") melts long too.

## One rule: the backing container decides

`Tables.columns` walks the tree, and at every node the *parent* decides the
orientation:

- a **`NamedTuple` record**, at any depth, melts each field to its **own wide
  column**;
- an **array/tuple-backed axis** melts **long** — one coordinate column per axis,
  plus a single `:value`.

```@example tab
using TreeArrays, Tables, Statistics, Random
Random.seed!(1)

X = TreeData(randn(500, 3, 5), :draw, :subject, :time => [0.0, 1.0, 2.0, 4.0, 8.0])
Y = mean(X; dims = :subject)

long = quantile(Y, TreeDim(:band, (0.1, 0.5, 0.9)); dims = :draw)
Tables.schema(long)
```

```@example tab
wide = quantile(Y, (lower = 0.1, median = 0.5, upper = 0.9); dims = :draw)
Tables.schema(wide)
```

Same data, same one-pass reduction; the only difference is whether you asked for
an axis or a record.

!!! warning "Named levels do not mean wide"
    A `TreeDim` whose levels are `Symbol`s is still an *axis*, and still melts
    **long**. Only a `NamedTuple` **parent** goes wide. This is precisely what
    `quantile(Y, :band => (lower = …, upper = …))` returns, and it melts into a
    long `:band` column.

Coordinate columns keep their level eltype — `String` levels give a `String`
column, `Symbol` levels a `Symbol` column, and an unlabelled axis gives a 1-based
`Int` position column. Nothing in the melt is numeric-specific.

## `TreeTable` — the view, and the pivot

[`TreeTable`](@ref) is the explicit tabular view. It is what you get implicitly
when you hand a `TreeData` to a Tables.jl consumer, and it takes one extra
argument the tree itself does not: `wide =`.

```@example tab
r = quantile(Y, :band => (lower = 0.1, median = 0.5, upper = 0.9); dims = :draw)

Tables.columnnames(Tables.columns(TreeTable(r)))               # long
```

```@example tab
Tables.columnnames(Tables.columns(TreeTable(r; wide = :band))) # pivoted
```

So orientation is a **consumer choice**, never a property of the data. The record
form bakes wide-ness in at reduction time; the axis + `wide =` form keeps it a
real axis and pivots only at the presentation boundary — which is what you want
when the same reduction feeds both a tidy long consumer and a ribbon plot, or
when you still need to reduce *over* the band axis.

### Column naming under `wide =`

- A **`Symbol`** level becomes a **bare** column (`:lower`). That is what makes the
  output drop straight into a `bands = [:lower => :upper]` plotting call.
- Any other level is **prefixed by its dim** — `:time` levels `[0.1, 0.25]` become
  `time_0_1`, `time_0_25` — so you never get a bare `0_1`.
- **Dots are rewritten to `_` everywhere**, `Symbol`s included. Vega-Lite reads
  `q0.025` as `datum["q0"]["025"]`, which would plot silently-wrong data rather
  than error.

Several wide dims are allowed: `wide = (:band, :param)` widens both, and the
columns are the cartesian product of their levels joined by `_` in `wide` order.
Column count multiplies and row count divides by each further axis's level count
— inherent to a pivot. `wide = :band` is what a ribbon wants; reach for `k > 1`
only when you genuinely want a fully-crossed table.

The rules, each of which errors by name:

- A wide dim may sit **anywhere in the melt**, not just at the top level — the
  band axis a chained reduction leaves on the *leaf* is the whole point.
- It must be a **real axis**; a fixed or ghost dim has no levels to spread.
- Naming the same dim twice errors.
- Colliding column names error — including collisions created *by* the dot
  rewriting, and (at `k > 1`) by the `_` join. `_` is not an injective separator:
  levels `(:x, :x_y)` × `(:y_z, :z)` would both give `x_y_z`, so it throws rather
  than mislabel.
- **Ragged throws under `wide =`, and only under `wide =`.** The long melt of the
  same tree works fine.

!!! note "Wide-mode column *names* are not type-inferable"
    Under `wide =`, the column names are the axis's coordinate *values*, which do
    not live in the type. The columns themselves are still concretely-typed lazy
    views, which is all a Tables.jl consumer needs. Long mode is unaffected and
    stays fully type-stable. Don't put a wide-mode `Tables.columns` call inside a
    hot type-stable kernel; call it at the plotting boundary.

## Laziness is the acceptance gate

`Tables.columns` assembles **lazy view columns** over the eagerly-computed
result. `@allocated` is flat across 100× rows, for homogeneous and heterogeneous
records alike:

```@example tab
n_rows(td) = length(Tables.columns(td).value)

small = TreeData(randn(10, 4), :draw, :time => [0.0, 1.0, 2.0, 3.0])
big   = TreeData(randn(1000, 4), :draw, :time => [0.0, 1.0, 2.0, 3.0])

(n_rows(small), n_rows(big))
```

```@example tab
Tables.columns(small); Tables.columns(big)     # warm up
(@allocated(Tables.columns(small)), @allocated(Tables.columns(big)))
```

Same number of bytes for 100× the rows: the columns are structure, not data.

Heterogeneous fields need no `Union` or box: a `(count::Int, mean::Float64)`
record becomes a typed `Int` column and a typed `Float64` column, each its own
concrete view. Metadata is never physically repeated — a constant dim is O(1).
Rows materialize only at the `rowtable` / DataFrame / JSON boundary.

## Plotting

Because the melt is a Tables.jl source, a plotting layer takes it directly:

```julia
using AlgebraOfVega   # or any Tables.jl-aware plotting layer

# long: one row per (x, level)
data(TreeTable(long)) * mapping(:time, :value; color = :band) * visual(:line)

# wide: bands as columns
data(TreeTable(r; wide = :band)) *
    mapping(:time, :median) * lineribbon(bands = [:lower => :upper])
```

Spaghetti plots work off the **unreduced** tree — a per-draw `String` id axis is a
first-class coordinate, so a grouping channel never fuses lines across cells:

```julia
draw_ids = ["$(combo)_$(d)" for d in 1:n_draw]
td = TreeData(vals, :draw => draw_ids, :time_h => times)   # no reduction at all
data(td) * mapping(:time_h, :value; group = :draw) * visual(:line)
```

The id column **is** the axis, and stays a lazy column — never materialized before
the serialization boundary.

## What the melt requires

Each of these errors clearly rather than producing a silently-wrong table:

- **A concrete tree type.** Sibling subtrees must share a *type*; they need not
  share a *shape*. What cannot melt is a tree whose axis lengths differ *as
  types* (a `:dose` axis of `(10, 20)` beside one of `(20,)` widens the container
  to a non-concrete eltype, so there is no static schema), and
  `TreeData(TreeData[], …)`, the empty container that erases its element type.
- **No `missing` in a leaf or record field.** A reduction can no longer produce
  one; only a hand-built `TreeData(:rec => (; a = 1, b = missing))` still can.
- **No raw, non-`TreeData` array-valued record field** — it carries no dim labels.
- Every real axis needs an enclosing array/tuple/record position to read a
  per-row coordinate from.

## [Feeding a numeric-array API](@id actual-array)

The Tables bridge feeds row/column consumers. A **numeric N-D-array** API —
`MCMCDiagnosticTools.ess`/`rhat`, or anything typed on `AbstractArray` — wants an
array, and a `TreeData` is deliberately not one. Three ways across, in preference
order.

**[`TreeActualArray(X)`](@ref)** — a lazy, **zero-copy** `AbstractArray{T,N}` view
that also keeps the dim labels. It works even when `parent(X)` is not an array,
which is the crucial case: draws held as a *record* of per-parameter
`(draw, chain)` matrices have a `NamedTuple` parent, and `TreeActualArray`
**promotes the record axis to a real array dimension**:

```@example tab
draws = TreeData(:param => (; mu = randn(500, 4), sigma = randn(500, 4)), :draw, :chain)
A = TreeActualArray(draws)
(size(A), A isa AbstractArray)
```

```@example tab
TreeArrays.dims(A)
```

```@example tab
A[1, 1, 2] === parent(draws).sigma[1, 1]    # zero-copy: it is the same element
```

A dense leaf `TreeData(arr3d, :draw, :chain, :param => names)` and that record
give the **same** array, so `ess(A)`/`rhat(A)` work whichever way the draws are
held. `parent(A)` recovers the tree, `dims(A)` its axes. Rectangular only —
ragged and heterogeneous shapes error by name.

**`parent(X)`** — the backing storage verbatim, zero-copy, but an `AbstractArray`
only when the leaf is array-backed, and it drops the labels. (A `selectdim` leaf's
parent is a `SubArray` — still an array, feeds fine.)

**`collect(X)` / `Array(X)`** — a materialized dense copy, for a consumer that
mutates or strictly requires an `Array`. `Array(X)` works if it can and
`MethodError`s otherwise.

## Rich display

A `TreeData` or `TreeTable` renders itself as HTML *and* as markdown, so it drops
straight into a web route, a Pluto notebook or a Jupyter cell with no wrapper:

- `TreeData` → a dim table (name / kind / coords) plus a bounded value preview. A
  record renders one collapsible section per field, recursing so each field shows
  its own dims; a ragged container previews its first few leaves.
- `TreeTable` → an actual `<table>` of the melt, in whatever orientation you asked
  for (a `wide =` table displays pivoted).

This lives in the core rather than an extension: `show`/`MIME`/`showable` are all
Base, so the methods cost nothing and need no plotting or web dependency.
TreeArrays escapes its own coordinates, so an axis labelled `"<b>"` cannot break
the DOM.

**Every preview is bounded and every elision is marked** (`…`, `N more`,
`N elements`) — currently 20 table rows, 3 ragged leaves, 8 tuple elements, 2000
characters per value dump. A display method must never densify the tree, and must
never look like the complete value. `string(td)` is a **preview**, not a
serialization: never parse it. If you want the data, use `Tables.columns` or
`parent(td)`.
