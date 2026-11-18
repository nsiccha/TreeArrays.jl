#!/usr/bin/env bash
# TreeArrays gordito one-off service launcher: runs the web harness
# (TreeArraysWeb, web/) in the foreground on loopback port 8122. The KB starts
# it as a one-off service (`POST /remote_service_start`, Restart, Switch to
# latest; skill expose-publicly §2b-0) and serves it at /p/TreeArrays/.
# It runs under non-interactive sshd, so PATH is set here.
# An optional first argument overrides the port, for a test start on a free
# scratch port; the KB passes none.
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.julia/juliaup/bin:/usr/local/bin:/usr/bin:/bin"
port=${1:-8122}
[[ "$port" =~ ^[0-9]+$ ]] || { echo "Error: port must be a number, got '$port'." >&2; exit 64; }
cd "$(dirname "$0")/../.."
# The KB resolves this environment on every start, restart and switch. Its
# Manifest is gitignored, so each start re-resolves it against the host's
# checkouts.
# kb-resolve: web/app
# web/app/main.jl binds 0.0.0.0 for the desktop's LAN access; this calls the
# serve function with the loopback host itself, and loads no Revise.
exec julia -t 2,1 --project=web/app -e '
    using TreeArraysWeb
    TreeArraysWeb.serve(; host="127.0.0.1", port=parse(Int, ARGS[1]), revise=nothing, async=true)
    wait()
' "$port"
