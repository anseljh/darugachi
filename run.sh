#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for file in "$root/.env" "$root/config.yaml"; do
  [[ -f "$file" ]] || { echo "Missing $file; run ./install.sh first." >&2; exit 1; }
done

set -a
. "$root/.env"
set +a

: "${API_KEY:?Set API_KEY in .env}"
: "${ADMIN_API_KEY:?Set ADMIN_API_KEY in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${LLAMA_SERVER:?Set LLAMA_SERVER in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
[[ -x "$LLAMA_SERVER" ]] || { echo "LLAMA_SERVER is not executable: $LLAMA_SERVER" >&2; exit 1; }
command -v docker >/dev/null || { echo "Install Docker with WSL integration first." >&2; exit 1; }
command -v python3 >/dev/null || { echo "Install Python 3 first." >&2; exit 1; }
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi >/dev/null

mkdir -p "$root/models.d"
python3 "$root/model_api.py" --models-dir "$root/models.d" --port "${MANAGE_PORT:-9293}" &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' EXIT INT TERM
"$root/bin/llama-swap" --config "$root/config.yaml" --config-dir "$root/models.d" --watch-config --listen "0.0.0.0:${PORT:-9292}"
