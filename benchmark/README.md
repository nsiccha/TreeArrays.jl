# Reduction and plot benchmarks

These scripts measure reductions on generated Gaussian arrays. They use no
external dataset or fitted model. Dimensions, probability levels, and the
operations being compared are declared in each script. They are developer
measurements and do not run during the documentation build.

`benchmarks.jl` compares a mean-over-subjects followed by quantiles-over-draws
pipeline. `plot/plot_benchmarks.jl` measures creation of plotting objects from
synthetic summaries. Their environments are `benchmark/Project.toml` and
`benchmark/plot/Project.toml`; develop the current TreeArrays checkout into each
environment before running the corresponding script.

Keep generated timings, figures, and environment manifests outside git. A
measurement used in a publication must include package revisions, Julia version,
thread count, workload dimensions, timing method, and machine capabilities.
No measurements from these scripts are asserted in the documentation.
