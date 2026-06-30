# Shared per-step benchmark helper for BOTH options (dev §4 — one core, no
# divergent near-copies). Measures BOTH wall-time and ALLOCATED BYTES: the
# differentiator ("never allocate the combined matrix" / "no padding waste")
# shows up in bytes far more clearly than sub-µs wall-time at this size.
module BenchKit

using BenchmarkTools

"""
    step(name, f; note="") -> NamedTuple

Warm up `f` once, then benchmark it. Returns `(; name, seconds, bytes, allocs,
note)`. Capped at `seconds=0.3` per step so a full 6-step × 2-option run stays
a few seconds (it's computed once and cached, not per request).
"""
function step(name, f; note="")
    f()                                   # warmup / compile
    tr = @benchmark $f() samples=200 seconds=0.3 evals=1
    (;
        name,
        seconds = minimum(tr.times) / 1e9,    # fastest sample, seconds
        bytes   = tr.memory,                  # allocated bytes per eval
        allocs  = tr.allocs,
        note,
    )
end

end # module BenchKit
