# TreeArrays

TreeArrays reduces named dimensions in nested arrays, including records and observations on different grids.

Install from a Julia-enabled shell:

```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/nsiccha/TreeArrays.jl")'
```

Average the draws at each observation time:

```@example start
using TreeArrays
X = TreeData([1.0 2.0; 3.0 4.0; 5.0 6.0], :draw, :time => [0, 1])
result = mean(X; dims=:draw)
(; time=coords(result, :time), average=vec(parent(result)))
```

The dimension name selects what to reduce. The result keeps the time labels.
The install command is a shell instruction; the example runs in the documentation environment.

## Next steps

| Task | Page |
| --- | --- |
| Construct and select labelled arrays or records | [Named axes and records](getting-started.md) |
| Summarize simulated concentration curves and plot their intervals | [Concentration trajectories](trajectories.md) |
| Choose a reducer or output layout | [Reductions](reductions.md) |
| Work with irregular observations | [Ragged data](ragged.md) |
| Look up a signature or restriction | [API](api.md) |
