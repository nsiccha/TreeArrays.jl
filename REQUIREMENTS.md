# TreeArrays.jl — Requirements, Features & Syntax

> Status: design draft, distilled from the `usage/main.jl` exploration. Syntax
> below is **illustrative** (current sketch + agreed direction), not final — the
> exact spelling is partly what the DD-exploration / mock-TA step exists to pin
> down.

## 1. Purpose & scope

A **postprocessing / summarization layer** for Bayesian posterior draws and
model predictions (PKPD context: subjects, doses, placebo, random effects,
draws/chains). Provides **named, hierarchically-structured, possibly-ragged
arrays** over flat backing storage, plus **streaming reductions** that compute
simple summary statistics.

- **In scope:** naming/selecting/reducing axes; assembling large logical arrays
  from blocks *without materializing* them; ragged (subject-varying-length) data
  and predictions; simple summary stats (mean / var / quantile / extrema /
  baseline-deltas).
- **Out of scope (upstream / external):** the heavy nonlinear compute — ODE
  solves, the model fit. TreeArrays consumes already-computed draws and produces
  summaries. Its compute is cheap-per-element but over potentially large arrays;
  the performance enemies are **allocation of the big combined matrix** and
  **per-block kernel quality**, *not* deferring expensive evaluations.

## 2. Relationship to DimensionalData (DD)

Heavy overlap (named dims, dim-aware reductions, selectors, broadcasting); reuse
DD's proven ideas (dims-as-types → compile-time axis resolution). Two deliberate
**differentiators**:

- **(a) Lazy/explicit *assembly* — NOT lazy compute.** Arithmetic is eager. What
  is lazy is *forming the big combined matrix*: the cross-product of
  categorical/virtual axes is described structurally and never allocated;
  reductions stream over it block-by-block. (DD's ops are eager *and* it
  materializes. Deferred `BroadcastArray`-style compute is explicitly avoided —
  the "lazy → bad perf" trap.)
- **(b) Ragged axes.** Subject-varying lengths (different #measurements / #doses
  per subject). DD is fundamentally rectangular.

**Build on DD vs standalone is the key open question** — the DD-exploration step
exists to answer it empirically.

## 3. Design principles

1. **Eager compute, lazy assembly.** Transforms (`constrain`, `zero_re`,
   `setdim`) and summary stats run *now*, in tight type-stable loops, on each
   materialized block. The combined matrix is structural; a reduction does
   *materialize one block → eager kernel → discard → next*. No deferred compute
   graph, no per-element closures, no recompute-on-reaccess.
2. **Structure eager, values transient.** Offsets / dims / meta are eager (shape
   is known without materializing). Assembled blocks are materialized on demand
   during a single-pass reduce and discarded. Escape hatch: `materialize(X)`
   pins a block when it is reused across several summaries.
3. **Factor is the core abstraction.** A *factor* = a categorical partition of an
   axis (a label per element ≡ index-sets per group). Factors can be **crossed**
   (sex × health), **nested** (measurement ⊂ subject), or overlapping.
   `mapslices(f; dims=:factor)` = group-by-factor + apply. A coarse factor
   **lifts** across a nesting via a join (each measurement inherits its
   subject's sex).
4. **Ragged = the contiguous-partition special case.** Group members contiguous
   in storage → offsets (CSR) → cheap contiguous views. Scattered membership
   (boolean masks, global sort) → label vector → **gather**. **Dense = uniform
   offsets.** One `mapslices` API, two kernels (view-loop / gather-loop).
   Physical storage can be clustered by **one** nested factor-sequence at a time
   (like a DB clustered index); "make X ragged" = "cluster storage by X".
5. **Dims are types (DD-style).** Dim *identity* (and small categorical *levels*)
   in the type → compile-time axis resolution + type stability. Value-lookups,
   ragged offsets, factor labels live in fields. High-cardinality levels stay in
   fields (avoid type bloat).

## 4. Core abstractions / types

- **`TreeArray(parent, meta)`** — flat backing `parent::AbstractArray` + `meta`
  describing axes / factors / fixed-coordinates.
- **Dim types** — one per dimension (`Draw`, `Param`, `Time`, `Dose`, `Subject`,
  `RandomEffect`, `Placebo`, `Space`, `QoI`, …); small categorical dims carry
  their levels in the type.
- **`(dim, level)` selector** — a *point on a dimension*, **dual**: used both to
  **subset** (`X[sel]` → the slice where dim == level) and as a **fixed
  coordinate** at construction (this array *is* the level-slice). One object,
  interpreted by the operation. This duality is what lets `stack` work.
  - sugar: `dim.level` for small categorical dims (compile-time-known);
    `dim(value)` / `At(value)` for value dims (runtime).
  - **Do not** overload `==` to return a selector (breaks the `==`→`Bool`
    contract); `dim.level` / `dim(value)` are the constructors.
- **Axis-lookup / factor** — dense (uniform offsets) | ragged (irregular
  offsets, contiguous) | scattered (label vector → gather). A ragged structure
  is a first-class, **identity-bearing** object multiple co-indexed columns
  share (e.g. `time` + `values`).

## 5. API surface (features + illustrative syntax)

```julia
# Construction — fixed coordinates given as (dim, level) selectors
S = TreeArray(randn(n_draws, n_cols),
        (; dims=(draw, param, random_effect.in_sample, placebo.on, space.sampler)))

# Set / fix a dimension (lazy override, no copy)
zero_re(X)    = setdim(X, random_effect.zero)
constrain(X)  = setdim(X, space.user)
setdose(X, d) = setdim(X, dose => d)

# Subsetting by a categorical level
X[random_effect.zero]

# Reductions over named dims — eager kernels; reducing a ragged axis re-densifies
mean(X; dims=:subject)
quantile(X, p; dims=:draw)

# mapslices — the workhorse; SAME code on rectangular AND ragged inputs
compute_stats(X) = mapslices(X; dims=:time) do L
    trough, peak  = extrema(L)
    baseline      = L[1]
    dtrough,dpeak = extrema(L .- baseline)
    (; trough, peak, baseline, dtrough, dpeak)
end

# Assembly — dense stack reads each input's fixed coord, promotes the differing dim
stack((S, S0))                                   # → random_effect ∈ (in_sample, zero)
stack(Iterators.product(doses, placebos)) do dose, placebo
    loc(setplacebo(setdose(P, dose), placebo))   # → dose, placebo axes
end
ragged_stack(per_subject_blocks)                 # incongruent lengths → ragged

materialize(X)   # escape hatch: pin a reused block
```

- **`mapslices` lowering:** `mapslices(f; dims=D)` ≡ `map(f, eachslice(...; over
  complement of D))` — views (no copy) + function barrier (specialize on `f`,
  concrete eltype) + collect a NamedTuple-returning `f` into a `StructArray`.
- **`stack` vs `ragged_stack` — two functions for type stability.** Dense-vs-
  ragged is a *runtime* property of block lengths; one polymorphic `stack` would
  return `Union{Dense,Ragged}`. `stack` asserts congruent blocks (errors → points
  at `ragged_stack`); raggedness stays **explicit / opt-in**.

## 6. The example data model (PKPD)

**Posterior draws `S`:** backing `n_draws × n_cols`; dims:
`draw` (groups: chains) · `param` (hierarchical: baseline → {fixed →
{intercept, zage, zweight, male, diseased}, log_scale, random[1:n_subjects]}) ·
`random_effect` ∈ {in_sample, zero, population} · `placebo` ∈ {on, off} ·
`space` ∈ {sampler, user}.

**Input data — one-to-many under `subject`** (length `n_subjects = 179`):
- per-subject **scalar factors**: `healthy`, `male` (Bool partitions),
  `weight`, `age` (continuous covariates) — dense over subject.
- ragged **measurement** series: `measurement_times → measurement_values`,
  ragged w.r.t. subject (shared offsets; `time` = lookup, `values` = entries).
- ragged **dose** history: `dose_times → dose_amounts`, ragged w.r.t. subject
  (a *distinct* offset structure from measurements).

**Prediction axes — the recurring `(ragged data, dense grid)` pattern** for any
quantity that is both *observed* and *swept*:
- `time`: ragged **obs-times** (data-conditional → residuals / PPC) + dense
  **grid** (counterfactual curves, cross-subject summaries).
- `dose`: ragged **received doses** (per-subject history) + dense **sweep**
  (`exp.(range(0,1,n_dose_levels))`, dose-response).

**`loc`:** (modified draws) → **rectangular predictions per block**. Ragged-vs-
dense is a property of the *assembled* time axis, not of `loc`.

## 7. Invariants

- **Reduce OVER a ragged axis re-densifies** (per-group → one value/group), so
  downstream `quantile(...; dims=subject)` works. **Reduce ACROSS the
  ragged-with axis** (cross-subject at a common time) → **dense grid only**.
- **`compute_stats` is seamless on rectangular and ragged** because rectangular
  = ragged-with-uniform-offsets; `mapslices(...; dims=:time)` hands `f` one
  contiguous per-group view either way.
- **Residuals** = obs block `.-` ragged-pred block, eager, on the **shared**
  ragged structure (pred carries an extra dense `draw` axis to broadcast over).

## 8. Open questions / decisions

1. **Build on DD vs standalone** — the DD-exploration step decides.
2. **Single-pass vs re-touch** — does anything reuse assembled intermediates
   across summaries (→ `materialize` escape hatch is load-bearing), or is
   everything a single-pass reduce?
3. **Levels-in-type cardinality threshold** — small categorical in the type;
   where is the cutoff to fields?
4. **`stack` ergonomics** — infer the new axis name/levels purely from inputs'
   fixed coords, or also accept an explicit dim?

## Appendix — cleanups already spotted in `usage/main.jl`

- `healty` → `healthy` typo (`:100`).
- `n_doses` name collision: ragged per-subject vector (`:41`) vs scalar sweep
  count (`:66`) → rename to `doses_per_subject` / `n_dose_levels`.
- Undefined helpers used in the sketch: `setre`, `loc`, `doses` / `placebos`,
  `population_quantiles` / `posterior_quantiles`, bare-name dims (`time`,
  `subject`, `draw`).
- Package core still unimplemented: `setdim`, `stack`, `mapslices` (stub
  `error()` at `:12`), `Statistics.var` for `TreeArray`.
