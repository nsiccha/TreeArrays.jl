# Threaded reductions must return what the serial loop returns, cell for cell -- same values
# (NaN cells included), same element types, same dims -- and fail the way it fails. Run by the
# "threaded reductions" test item, in a multi-threaded child process when the suite itself has one
# thread (threads.jl), because with one thread the threaded path never runs.
using Test, TreeArrays, NaNStatistics, Statistics

@assert Threads.nthreads() > 1 "threads_checks.jl must run with more than one thread"

# Identity, not approximate equality: a cell computed on another task is the same computation.
_cells(X::TreeData) = _cells(parent(X))
_cells(p::AbstractArray) = map(_cells, p)
_cells(p::Tuple) = map(_cells, p)
_cells(p::NamedTuple) = map(_cells, p)
_cells(v) = v
function same(a::TreeData, b::TreeData)
    typeof(a) == typeof(b) && isequal(TreeArrays.dims(a), TreeArrays.dims(b)) && isequal(_cells(a), _cells(b))
end

const Q   = (lower = 0.025, median = 0.5, upper = 0.975)
const PD  = TreeDim(:pct, (0.1, 0.5, 0.9))
const ts  = collect(0.0:0.5:99.5)

# (time x qoi x draw): 200 x 3 x 400 = 240_000 elements, enough work for several tasks.
x = randn(200, 3, 400)
x[rand(eachindex(x), 5_000)] .= NaN          # scattered NaNs...
x[7, 2, :] .= NaN                             # ...and one all-NaN slice
X  = TreeData(x, :time => ts, :qoi => (:a, :b, :c), :draw => Base.OneTo(400))
Xc = TreeData(randn(200, 3, 400), :time => ts, :qoi => (:a, :b, :c), :draw => Base.OneTo(400))

@testset "the checks below really take the threaded path" begin
    @test TreeArrays._ntasks(600, length(x), true) >= 2
    @test TreeArrays._ntasks(600, length(x), false) == 1
end

@testset "built-in reducers: threaded == serial, cell for cell" begin
    for (name, f) in (
            "nanquantile pdim"  => (X; kw...) -> nanquantile(X, PD; dims = :draw, kw...),
            "nanquantile record" => (X; kw...) -> nanquantile(X, Q; dims = :draw, kw...),
            "nanquantile band"  => (X; kw...) -> nanquantile(X, :band => Q; dims = :draw, kw...))
        @testset "$name" begin
            @test same(f(X), f(X; threads = false))
        end
    end
    for (name, f) in (
            "quantile pdim"   => (X; kw...) -> quantile(X, PD; dims = :draw, kw...),
            "quantile record" => (X; kw...) -> quantile(X, Q; dims = :draw, kw...),
            "quantile band"   => (X; kw...) -> quantile(X, :band => Q; dims = :draw, kw...),
            "hdi"             => (X; kw...) -> hdi(X; dims = :draw, mass = 0.8, kw...),
            "mean"            => (X; kw...) -> mean(X; dims = :draw, kw...),
            "sum"             => (X; kw...) -> sum(X; dims = :draw, kw...),
            "var"             => (X; kw...) -> var(X; dims = :draw, kw...),
            "std pooled"      => (X; kw...) -> std(X; dims = (:draw, :time), corrected = false, kw...))
        @testset "$name" begin
            @test same(f(Xc), f(Xc; threads = false))
        end
    end
end

@testset "chained reduction (:subject, then :draw) -- record slices" begin
    x4 = randn(20, 50, 3, 1000)               # subject x time x qoi x draw
    X4 = TreeData(x4, :subject, :time => ts[1:50], :qoi => (:a, :b, :c), :draw)
    @test TreeArrays._ntasks(150, 50 * 3 * 1000, true) >= 2   # the second reduce threads too
    chain(; kw...) = nanquantile(nanquantile(X4, (p05 = 0.05, p50 = 0.5); dims = :subject, kw...), Q; dims = :draw, kw...)
    @test same(chain(), chain(threads = false))
    @test parent(chain()) isa Matrix
end

@testset "ragged outer-axis gather (array-backed leaves)" begin
    R = TreeData([TreeData(randn(1000), :time) for _ in 1:100], :draw)
    @test TreeArrays._ntasks(1000, 1000 * 100, true) >= 2
    @test same(nanquantile(R, Q; dims = :draw), nanquantile(R, Q; dims = :draw, threads = false))
    @test same(mean(R; dims = :draw), mean(R; dims = :draw, threads = false))
end

@testset "pooled straddle (:draw, :chain) keeps one bag per task" begin
    C = TreeData([TreeData(randn(500 + 10j, 60), :draw, :param) for j in 1:4], :chain)
    @test same(nanquantile(C, Q; dims = (:draw, :chain)), nanquantile(C, Q; dims = (:draw, :chain), threads = false))
    @test same(std(C; dims = (:draw, :chain)), std(C; dims = (:draw, :chain), threads = false))
end

@testset "mapslices: serial unless asked; opt-in threads a user kernel" begin
    seen = Task[]
    mapslices(Xc; dims = :draw) do s
        push!(seen, current_task())            # safe only because the default IS serial
        sum(s)
    end
    @test all(==(current_task()), seen) && length(seen) == 600
    k(s) = (; lo = minimum(s), hi = maximum(s))
    @test same(mapslices(s -> TreeData(:r => k(s)), Xc; dims = :draw, threads = true),
               mapslices(s -> TreeData(:r => k(s)), Xc; dims = :draw))
    # coords = true binds per node and still threads
    auc(y, t) = sum(i -> (t[i+1] - t[i]) * (y[i+1] + y[i]) / 2, 1:length(t)-1)
    @test same(mapslices(auc, Xc; dims = :time, coords = true, threads = true),
               mapslices(auc, Xc; dims = :time, coords = true))
end

@testset "a kernel whose output type varies widens exactly like Base's map" begin
    flip(s) = s[1] > 0 ? 1 : 1.0               # Int on some slices, Float64 on others
    a = mapslices(flip, Xc; dims = :draw, threads = true)
    b = mapslices(flip, Xc; dims = :draw)
    @test typeof(parent(a)) == typeof(parent(b)) == Matrix{Real}
    @test same(a, b)
end

@testset "a failing kernel throws the serial loop's exception, unwrapped" begin
    Xn = TreeData(copy(parent(Xc)), :time => ts, :qoi => (:a, :b, :c), :draw => Base.OneTo(400))
    parent(Xn)[150, 1, 9] = NaN; parent(Xn)[30, 3, 1] = NaN    # two failing slices, far apart
    serial   = try hdi(Xn; dims = :draw, threads = false); nothing catch e; e end
    threaded = try hdi(Xn; dims = :draw); nothing catch e; e end
    @test serial isa ArgumentError && threaded isa ArgumentError
    @test serial.msg == threaded.msg
    # the EARLIEST failing slice in serial order wins, whichever task finishes first
    bad(s) = (i = findfirst(>(3.3), s); isnothing(i) ? 0.0 : throw(DomainError(s[i], "slice")))
    es = try mapslices(bad, Xc; dims = :draw); nothing catch e; e end
    et = try mapslices(bad, Xc; dims = :draw, threads = true); nothing catch e; e end
    @test es isa DomainError && et isa DomainError && es.val === et.val
end
