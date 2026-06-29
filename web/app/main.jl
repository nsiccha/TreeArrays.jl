using Revise
using TreeArraysWeb

begin
    TreeArraysWeb.terminate()
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8099
    TreeArraysWeb.serve(; host="0.0.0.0", revise=:lazy, port, async=true)
end
