# Julia >= 1.12 reads the pinned [sources] entries in docs/Project.toml.
using Pkg
VERSION >= v"1.12" || error("Documentation setup needs Julia 1.12 or newer for [sources] support.")
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()
