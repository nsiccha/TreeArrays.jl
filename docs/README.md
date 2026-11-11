# Documentation

The navigation has four groups, in this order. Each page has one purpose:

| Group | Page | Purpose |
| --- | --- | --- |
| Start | `index.md` | Install TreeArrays and reduce a labelled array. |
| Guides | `getting-started.md` | Construct, select, and broadcast over named axes and records. |
| Guides | `reductions.md` | Reduce named dimensions and choose a summary's output structure. |
| Guides | `ragged.md` | Reduce irregular grids and choose coordinate alignment. |
| Guides | `streaming.md` | Retain per-draw statistics and reopen a mapped cache. |
| Guides | `tables.md` | Choose a table or array view and convert external containers. |
| Examples | `trajectories.md` | Summarize and plot simulated trajectories on irregular grids. |
| Reference | `api.md` | Look up public types, operations, and their restrictions. |

Runnable Julia blocks execute during `julia --project=docs docs/make.jl`.
On a fresh checkout, use Julia 1.12 or newer and run
`julia --project=docs docs/instantiate.jl` first. It reads the pinned public
package sources in `docs/Project.toml`; no generated Manifest is committed.
Python 3 is required for the local lint. Managed fleet builds use the canonical
`kb-docs-build` helper, which also supports the host's Julia 1.10 environment.
The build fails on documentation errors and checks exported docstrings.
`python3 docs/lint.py` checks the page list, prose, code fences, and example
structure. Generated figures, build receipts, caches, and dependency manifests
belong in the ignored build/environment directories, never in git.

The example uses synthetic draws, not a fitted model. Its Reproduce footer
records the Julia/package versions and a source digest from the build. No fit
or benchmark runs in CI. The former `comparisons.md` essay and unbuilt
`pkpd_demo.jl` are removed; the supported bridges live in `tables.md`.
