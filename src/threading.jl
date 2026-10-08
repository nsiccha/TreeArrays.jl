# ===================== threaded slice loops =====================
# A reduction's slices are independent: each kernel call reads ONE slice and produces ONE output
# cell. So the hot loops of the reduction machinery (`_reduceouter`'s kept-index map, the ragged
# gather, the pooled straddle -- mapslices.jl) may run their iterations on several tasks and still
# return the SAME array, cell for cell: every cell is computed by the same kernel code on the same
# slice and written to its own position. Nothing is combined ACROSS slices, so there is no
# floating-point reassociation that could make a threaded result differ from a serial one.
# (snag nanquantile-redu: an 8067-slice `nanquantile` took the same 1.26 s at 1 and 8 threads.)
#
# What is NOT independent is kernel STATE. TreeArrays' own slice kernels reuse one working buffer
# across slices (`_ScratchKernel` below); two tasks sharing it would sort each other's data. So every
# task runs its own copy of the kernel (`_taskcopy`), and a kernel instance is only ever used by one
# task at a time. A plain function carries no state TreeArrays can see, so `_taskcopy` hands it back
# as is: running an arbitrary kernel threaded is the CALLER's assertion that it is task-safe, which
# is why `mapslices` only threads on request (`threads = true`) while TreeArrays' own reducers, whose
# kernels it knows, thread by default.

# A slice kernel that needs a private, reused working buffer: `quantile!` sorts in place, the NaN
# compaction partitions in place, `hdi` sorts. The buffer belongs to the kernel INSTANCE -- one per
# call on the serial path (exactly the old call-local `scratch`), one per task on the threaded path.
# Entry re-fills it from the slice, so whatever the previous slice left there (a shrunken length
# after NaN compaction, a sorted order) is never visible to the next.
struct _ScratchKernel{F,T}
    f::F
    scratch::Vector{T}
end
_ScratchKernel(f, ::Type{T}) where T = _ScratchKernel(f, T[])
function (k::_ScratchKernel)(slice)
    s = k.scratch
    length(s) == length(slice) || resize!(s, length(slice))
    copyto!(s, slice)
    k.f(s)
end

# A fresh, task-private instance of a kernel. Stateful kernel types say how to copy themselves;
# everything else is shared by reference (see the note above).
_taskcopy(f) = f
_taskcopy(k::_ScratchKernel) = _ScratchKernel(k.f, similar(k.scratch, 0))

# Below this many elements of work per task, spawning costs more than it saves -- a `sum` over a few
# thousand numbers is microseconds, a task round trip is not free. A slice-sort kernel (`quantile`)
# clears the bar long before a streaming one (`mean`) does, but both are measured in ELEMENTS
# touched, which is what this counts.
const _TASK_MIN_WORK = 1 << 15

# How many tasks to split `n` independent iterations (about `work` elements in all) into. Element 1
# is always computed on the calling task first (see `_tmap`), so `n - 1` is the most that can run
# concurrently. 2x the default pool leaves room to balance uneven slices (NaN-heavy ones are
# cheaper) and a pool already busy with other work. One task would only add a round trip.
_ntasks(n, work, threads::Bool) =
    threads && Threads.nthreads() > 1 ? min(n - 1, 2 * Threads.nthreads(), work ÷ _TASK_MIN_WORK) : 1

# `map(x -> body(k, x), xs)`, possibly on several tasks, each with its own `_taskcopy(k)`. Returns
# exactly what the serial `map` returns -- same cells, same element type, same shape -- so callers
# cannot tell the two paths apart:
#   * cell 1 runs first, on the calling task, and fixes the element type; every other cell is
#     written in place by the task that owns its chunk;
#   * a chunk that produces a cell of a DIFFERENT type stops, and the whole map re-runs serially --
#     that is the only way to reproduce Base's element-type widening exactly, and only a kernel
#     whose return type varies with the data ever pays for it;
#   * a kernel that throws rethrows the exception of the EARLIEST failing chunk, unwrapped -- each
#     chunk stops at its own first failure, so that is the same cell, and the same exception, the
#     serial loop would have thrown.
function _tmap(body::B, k, xs::AbstractArray, work::Integer, threads::Bool) where B
    n  = length(xs)
    nt = _ntasks(n, work, threads)
    nt < 2 && return map(x -> body(k, x), xs)
    cis = vec(CartesianIndices(xs))
    x1  = body(k, xs[cis[1]])
    out = similar(xs, typeof(x1))
    out[cis[1]] = x1
    chunks = collect(Iterators.partition(2:n, cld(n - 1, nt)))
    tasks  = map(chunks) do r
        kk = _taskcopy(k)              # copied HERE, on the calling task: `k` is never shared
        Threads.@spawn _tmapchunk!(out, body, kk, xs, cis, r)
    end
    _tmapsync(tasks) && return out
    map(x -> body(k, x), xs)           # some cell's type differed from cell 1's: widen exactly as Base
end

# A function barrier, so each task's loop specializes on the kernel and the output element type.
function _tmapchunk!(out::AbstractArray{T}, body::B, k, xs, cis, r) where {T,B}
    for j in r
        I = cis[j]
        v = body(k, xs[I])
        v isa T || return false
        out[I] = v
    end
    true
end

# Wait for EVERY task before deciding anything (none is left running behind the caller's back),
# then surface the earliest chunk's failure as the kernel's own exception. A `TaskFailedException`
# is only caught to be inspected below -- never dropped.
function _tmapsync(tasks)
    for t in tasks
        try
            wait(t)
        catch e
            e isa TaskFailedException || rethrow()
        end
    end
    i = findfirst(istaskfailed, tasks)
    isnothing(i) || throw(tasks[i].exception)
    all(fetch, tasks)
end
