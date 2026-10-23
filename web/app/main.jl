using TreeArraysWeb

begin
    TreeArraysWeb.terminate()
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8099
    TreeArraysWeb.serve(; host="0.0.0.0", revise=nothing, port, async=true)
end
