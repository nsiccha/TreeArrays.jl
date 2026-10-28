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
