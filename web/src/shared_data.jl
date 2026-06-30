# Shared synthetic PKPD input for BOTH option implementations, so the
# DD-vs-mock comparison is apples-to-apples. Seeded → reproducible numbers and
# timings. Distilled from `usage/main.jl:34–66` with the Appendix fixes
# (`healthy` typo, `n_doses` collision → `doses_per_subject`/`n_dose_levels`).
module SharedData

using Random

"""
    synthetic_data(; n_subjects=179, n_draws=1000, seed=42)

Returns a NamedTuple with:
- posterior draws `draws::Matrix` (`n_draws × n_cols`) + `re_cols` = the
  random-effect column range within the `baseline` param group (the columns
  `zero_re` zeros);
- per-subject dense scalar factors (`healthy`/`male`/`weight`/`age`);
- ragged dose history (`dose_times`/`dose_amounts`, 1–28 per subject);
- ragged measurement series (`measurement_times`/`measurement_values`, 2–100
  per subject).

`n_cols` follows the sketch's `n_params` layout (≈ 380 for 179 subjects) so the
stacked combined block is the ~6 MB the design's "never allocate the combined
matrix" claim is about.
"""
function synthetic_data(; n_subjects=179, n_draws=1000, seed=42)
    rng = Xoshiro(seed)

    # Per-subject dense factors.
    healthy = rand(rng, Bool, n_subjects)
    male    = rand(rng, Bool, n_subjects)
    weight  = randn(rng, n_subjects)
    age     = randn(rng, n_subjects)

    # Ragged dose history (distinct offset structure from measurements).
    doses_per_subject = [rand(rng, 1:28) for _ in 1:n_subjects]
    dose_times   = [sort(randn(rng, k)) for k in doses_per_subject]
    dose_amounts = [randn(rng, k)       for k in doses_per_subject]

    # Ragged measurement series.
    n_measurements     = [rand(rng, 2:100) for _ in 1:n_subjects]
    measurement_times  = [sort(randn(rng, k)) for k in n_measurements]
    measurement_values = [randn(rng, k)       for k in n_measurements]

    # Posterior draws. Param layout (sketch §6 / usage/main.jl:55–66):
    #   baseline group: fixed 1:5, log_scale 6, random 7:6+n_subjects
    #   + dose / placebo / effect / noise / other blocks → n_cols ≈ 380.
    n_params = (;
        baseline = 6 + n_subjects,
        dose     = 6 + n_subjects,
        placebo  = 3,
        effect   = 4,
        noise    = 1,
        other    = 2,
    )
    n_cols  = sum(n_params)
    re_cols = 7:(6 + n_subjects)          # random-effect columns (what zero_re zeros)
    draws   = randn(rng, n_draws, n_cols)

    (;
        n_subjects, n_draws, n_cols, re_cols,
        healthy, male, weight, age,
        doses_per_subject, dose_times, dose_amounts,
        n_measurements, measurement_times, measurement_values,
        draws,
    )
end

end # module SharedData
