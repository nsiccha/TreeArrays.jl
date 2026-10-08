@testitem "threaded reductions match serial cell for cell" begin
    using Test
    using TreeArrays
    # The threaded path only runs with more than one thread, and CI / a default `Pkg.test` start
    # ONE -- so re-run the checks in a 4-thread child process rather than pass them vacuously.
    script = joinpath(pkgdir(TreeArrays), "test", "threads_checks.jl")
    if Threads.nthreads() > 1
        include(script)
    else
        cmd = `$(Base.julia_cmd()) --threads=4 --startup-file=no --project=$(Base.active_project()) $script`
        @test success(pipeline(cmd; stdout, stderr))
    end
end
