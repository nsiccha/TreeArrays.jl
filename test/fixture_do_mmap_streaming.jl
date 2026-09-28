# Run in a consumer environment with this TreeArrays candidate and the matching
# DynamicObjects candidate. No fits, external data, or full trajectory caches.
using Test, TreeArrays, DynamicObjects, Statistics
using TreeArrays: meta, name

const MMAP = Val(:mmap)

function assert_same(a::TreeData, b::TreeData)
    assert_same(meta(a), meta(b))
    assert_same(parent(a), parent(b))
end
assert_same(a::TreeDim, b::TreeDim) = (a_name = name(a); @test a_name == name(b); assert_same(meta(a), meta(b)))
function assert_same(a::NamedTuple, b::NamedTuple)
    @test keys(a) == keys(b)
    foreach(assert_same, values(a), values(b))
end
function assert_same(a::Tuple, b::Tuple)
    @test length(a) == length(b)
    foreach(assert_same, a, b)
end
function assert_same(a::AbstractArray, b::AbstractArray)
    @test size(a) == size(b)
    foreach(assert_same, a, b)
end
assert_same(a, b) = @test isequal(a, b)

# Numeric arrays returned by the loader point into a file mapping on Linux.
# Inspect the address, never write to a readonly cached array to prove it.
function assert_mapped(a::Array)
    isempty(a) && return
    address = UInt(pointer(a))
    mappings = readlines("/proc/self/maps")
    @test any(mappings) do line
        bounds = split(first(split(line)), '-'; limit=2)
        lo, hi = parse.(UInt, bounds; base=16)
        lo <= address < hi && occursin("fixture", line) && split(line)[2][2] != 'w'
    end
end
assert_mapped(x::TreeData) = assert_mapped(parent(x))
assert_mapped(x::Union{Tuple,NamedTuple}) = foreach(assert_mapped, x)
assert_mapped(x::AbstractArray{<:TreeData}) = foreach(assert_mapped, x)
assert_mapped(x::Array{<:TreeData}) = foreach(assert_mapped, x)
assert_mapped(x::Number) = nothing

function layout_cases()
    dense = TreeData(reshape(collect(1.0:12.0), 4, 3),
        :draw => (1, 2, 3, 4), :time => range(0, 1; length=3); dose=20)
    record = TreeData(:stat => (
        total=TreeData(collect(1.0:4.0), :draw => [1,2,3,4]),
        flag=Bool[true,false,true,false],
        complex=ComplexF64[1+2im, 3+4im, 5+6im, 7+8im],
        scalar=2.5), :draw => [1,2,3,4])
    ragged = TreeData([TreeData(collect(1.0:n), :time => collect(1:n))
        for n in (0, 2, 5)], :subject => ("empty", "short", "long"))
    proto = parent(ragged)[2]
    empty = TreeData(typeof(proto)[], :subject => String[])
    positional = TreeData((dense, record), TreeDim(:scenario, (:a,:b)))
    ghosts = mapslices(sum, dense; dims=:time)
    viewleaf = TreeData(view(parent(dense), 1:2:4, :),
        :draw => (1,3), :time => range(0,1; length=3))
    (;dense, record, ragged, empty, positional, ghosts, viewleaf)
end

@dynamicstruct struct StreamingFixture
    cache_dir
    draw_labels
    ntime
    __cache_base__ = cache_dir
    @mmap v"1" compact(generator, statistics)::TreeData = map(TreeDim(:draw, draw_labels)) do draw
        trajectory = generator(draw, ntime)
        TreeData(:stat => map(f -> fill(f(trajectory)), statistics))
    end
    @mmap v"1" unannotated(generator, statistics) = compact(generator, statistics)
    @mmap v"1" full(generator) = map(TreeDim(:draw, draw_labels)) do draw
        TreeData(generator(draw, ntime), :time => 1:ntime)
    end
end

function public_trajectory(draw, ntime)
    first(ARGS) == "reopen" && error("cold reopen unexpectedly invoked the generator")
    # The parent process counts these receipts: exactly one call per draw.
    println("SIMULATION ", draw)
    [Float64(draw) + j / ntime for j in 1:ntime]
end
public_sum(y) = (println("STAT total"); sum(y))
threshold_stat(threshold) = y -> (println("STAT above"); count(v -> v > threshold, y))

function stream_child(mode, directory)
    job = StreamingFixture(directory, [1,2,3,4], 1024)
    stats = (;total=public_sum, above=threshold_stat(2.5))
    X = job.compact(public_trajectory, stats)
    originals = [joinpath(root, file) => read(joinpath(root, file))
        for (root, _, files) in walkdir(directory) for file in files]
    @test coords(X, :draw) == [1,2,3,4]
    @test propertynames(X) == (:total, :above)
    # Independent closed-form oracle; never invoke the trajectory generator.
    for (draw, record) in zip(1:4, parent(X))
        @test parent(record.total)[] == 1024draw + 1025/2
        @test parent(record.above)[] == clamp(1024 - floor(Int, 1024(2.5-draw)), 0, 1024)
    end
    assert_mapped(X)
    median_leaf = only(parent(quantile(X.total, TreeDim(:q, 0.5); dims=:draw)))
    @test parent(median_leaf) == 3072.5
    @test sum(parent(record.total)[] * parent(record.above)[] for record in parent(X)) == 9700608
    if mode == "write"
        # Function bundles and captures participate in the same ordinary DO key.
        assert_same(X, job.compact(public_trajectory, stats))
        changed = job.compact(public_trajectory, (;total=public_sum, above=threshold_stat(3.5)))
        @test parent(parent(changed)[3].above)[] == 512
        subset = job.compact(public_trajectory, (;total=public_sum))
        @test propertynames(subset) == (:total,)
        @test propertynames(X) == (:total, :above) # earlier result retained
        @test parent(parent(X)[3].above)[] == 1024
        assert_same(X, job.unannotated(public_trajectory, stats))
        configured = StreamingFixture(directory, [1,2,3,4], 512)
        different = configured.compact(public_trajectory, stats)
        @test parent(parent(different)[1].total)[] == 768.5
        for (path, bytes) in originals
            @test read(path) == bytes
        end
    end
    println("CHILD_OK ", mode)
end

function fixture_child(::Val{:layouts}, directory)
    for (label, expected) in pairs(layout_cases())
        restored = DynamicObjects.load(MMAP, joinpath(directory, "fixture-$label.mmap"), TreeData)
        assert_same(expected, restored)
        assert_mapped(restored)
    end
    println("LAYOUTS_COLD_OK")
end
fixture_child(::Val{:write}, directory) = stream_child("write", directory)
fixture_child(::Val{:reopen}, directory) = stream_child("reopen", directory)

function run_parent()
    @test !isnothing(Base.get_extension(TreeArrays, :TreeArraysDynamicObjectsExt))
    mktempdir(prefix="fixture-do-mmap-") do directory
        @testset "named numeric layouts, readonly mmap and retained inputs" begin
            for (label, original) in pairs(layout_cases())
                snapshot = deepcopy(original)
                path = joinpath(directory, "fixture-$label.mmap")
                DynamicObjects.save(MMAP, path, original)
                restored = DynamicObjects.load(MMAP, path, TreeData)
                assert_same(original, restored)
                assert_same(original, snapshot)
                assert_mapped(restored)
                # Registration also routes unannotated reads to the extension.
                assert_same(restored, DynamicObjects.load(MMAP, path))
                if label == :dense
                    @test typeof(restored) == typeof(original)
                    assert_same(restored, DynamicObjects.load(MMAP, path, typeof(original)))
                end
                if label == :viewleaf
                    @test_throws ArgumentError DynamicObjects.load(MMAP, path, typeof(original))
                end
            end
            fixture = abspath(@__FILE__)
            project = dirname(Base.active_project())
            cold = read(`$(Base.julia_cmd()) --startup-file=no --project=$project $fixture layouts $directory`, String)
            @test occursin("LAYOUTS_COLD_OK", cold)
        end
        @testset "ragged semantics survive persistence" begin
            X = layout_cases().ragged
            path = joinpath(directory, "fixture-ragged.mmap")
            DynamicObjects.save(MMAP, path, X)
            Y = DynamicObjects.load(MMAP, path, TreeData)
            assert_same(mapslices(sum, X; dims=:time), mapslices(sum, Y; dims=:time))
            @test coords(parent(Y)[3], :time) == collect(1:5)
            @test size(parent(Y)[1]) == (0,)
        end
        @testset "unsupported and incomplete containers fail" begin
            bad = TreeData(["unsupported"], :draw)
            err = try
                DynamicObjects.save(MMAP, joinpath(directory, "fixture-bad.mmap"), bad)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("String", sprint(showerror, err))
            @test_throws ArgumentError DynamicObjects.save(MMAP,
                joinpath(directory, "fixture-scalar-bad.mmap"), TreeData("unsupported"))
            path = joinpath(directory, "fixture-truncated.mmap")
            DynamicObjects.save(MMAP, path, layout_cases().dense)
            bytes = read(path)
            write(path, bytes[1:end-1]) # fresh path, with no escaped mapped value
            @test_throws ArgumentError DynamicObjects.load(MMAP, path, TreeData)
            write(path, [bytes; 0x00])
            @test_throws ArgumentError DynamicObjects.load(MMAP, path, TreeData)
        end
        @testset "callback-retained trajectories are never reused" begin
            retained = map(TreeDim(:draw, [1,2,3,4])) do draw
                trajectory = [Float64(draw) + j/16 for j in 1:16]
                before = copy(trajectory)
                # Every returned argument/view remains readable after later draws.
                result = TreeData(:stat => (
                    total=fill(sum(trajectory)), above=fill(count(>(2.5), trajectory))))
                TreeData(:retained => (;trajectory, before, result,
                    firsthalf=view(trajectory, 1:8)))
            end
            for record in parent(retained)
                @test parent(record.trajectory) == parent(record.before)
                @test parent(record.firsthalf) == parent(record.before)[1:8]
                @test parent(record.result.total)[] == sum(parent(record.before))
            end
        end
        @testset "separate-process serial producer and cache identity" begin
            cache = joinpath(directory, "fixture-cache")
            fixture = abspath(@__FILE__)
            project = dirname(Base.active_project())
            writer = read(`$(Base.julia_cmd()) --startup-file=no --project=$project $fixture write $cache`, String)
            lines = split(writer, '\n')
            @test count(==("SIMULATION 1"), lines) == 4
            @test count(l -> startswith(l, "SIMULATION "), lines) == 16
            @test lines[1:12] == vcat([ ["SIMULATION $draw", "STAT total", "STAT above"] for draw in 1:4 ]...)
            @test occursin("CHILD_OK write", writer)
            reader = read(`$(Base.julia_cmd()) --startup-file=no --project=$project $fixture reopen $cache`, String)
            @test !occursin("SIMULATION ", reader)
            @test occursin("CHILD_OK reopen", reader)
            # Compare complete retained files against an explicit full request.
            compact_bytes = sum(filesize(joinpath(root, file)) for (root, _, files) in walkdir(cache) for file in files)
            full_path = joinpath(directory, "fixture-full.mmap")
            full = map(TreeDim(:draw, [1,2,3,4])) do draw
                TreeData([Float64(draw) + j/1024 for j in 1:1024], :time => 1:1024)
            end
            DynamicObjects.save(MMAP, full_path, full)
            @test compact_bytes < filesize(full_path)
            println("PAYLOAD_BYTES compact_all_requests=", compact_bytes,
                " full_one_request=", filesize(full_path), " full_numeric=", 4*1024*8,
                " compact_numeric_per_request=", 4*2*8)
        end
    end
end

if isempty(ARGS)
    run_parent()
else
    fixture_child(Val(Symbol(ARGS[1])), ARGS[2])
end
