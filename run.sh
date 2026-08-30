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

# Every vllm-pooling definition substitutes these, so they must always be set --
# including when VLLM_MODEL is unset and the only vLLM models are ones the admin
# API registered. llama-swap's ${env.X} has no default syntax: unset would
# substitute empty and vLLM would die on a valueless flag.
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.75}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-10240}
export VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN
command -v python3 >/dev/null || { echo "Install Python 3 first." >&2; exit 1; }

mkdir -p "$root/models.d"
defaults="$root/models.d/_defaults.yaml"
tmp=$(mktemp "$root/models.d/.defaults.XXXXXX")

llama_enabled=false
if [[ -n "${LLAMA_HF_REPO:-}" ]]; then
  : "${LLAMA_SERVER:?Set LLAMA_SERVER when LLAMA_HF_REPO is set}"
  [[ -x "$LLAMA_SERVER" ]] || { echo "LLAMA_SERVER is not executable: $LLAMA_SERVER" >&2; exit 1; }
  llama_enabled=true
else
  echo "Skipping default llama.cpp model: LLAMA_HF_REPO is unset."
fi

vllm_enabled=false
if [[ -n "${VLLM_MODEL:-}" && -n "${VLLM_IMAGE:-}" ]]; then
  command -v docker >/dev/null || { echo "Install Docker with WSL integration first." >&2; exit 1; }
  HF_HOME=${HF_HOME:-"$root/hf-cache"}
  export HF_HOME
  mkdir -p "$HF_HOME"
  echo "Checking Docker GPU access (the first run downloads a CUDA image)..."
  docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
  vllm_enabled=true
else
  echo "Skipping default vLLM model: VLLM_MODEL and VLLM_IMAGE must both be set."
fi

{
  if "$llama_enabled" || "$vllm_enabled"; then
    printf '%s\n' 'models:'
  else
    printf '%s\n' 'models: {}'
  fi
  if "$llama_enabled"; then
    cat <<'EOF'
  llama-reranker:
    cmd: >-
      ${env.LLAMA_SERVER} --host 127.0.0.1 --port ${PORT}
      --hf-repo ${env.LLAMA_HF_REPO} --hf-token ${env.HF_TOKEN}
      --embedding --pooling rank --reranking --n-gpu-layers all
    capabilities:
      reranker: true
EOF
  fi
  if "$vllm_enabled"; then
    cat <<'EOF'
  vllm-embed:
    cmdStop: docker stop lan-vllm-embed
    cmd: >-
      docker run --init --rm --name lan-vllm-embed --gpus all
      -e HF_TOKEN=${env.HF_TOKEN}
      -v ${env.HF_HOME}:/root/.cache/huggingface
      -p ${PORT}:8000 ${env.VLLM_IMAGE}
      --model ${env.VLLM_MODEL} --served-model-name vllm-embed --runner pooling
EOF
  fi
} > "$tmp"
mv "$tmp" "$defaults"

# Older generated config files carried default models. The admin API now owns
# model definitions, so preserve only the global settings before `models:`.
runtime_config="$root/.config.runtime.yaml"
awk '/^models:/{exit} {print}' "$root/config.yaml" > "$runtime_config"
echo "Starting controller on ports ${PORT:-9292} and ${MANAGE_PORT:-9293}..."
python3 "$root/model_api.py" --models-dir "$root/models.d" --port "${MANAGE_PORT:-9293}" &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' EXIT INT TERM
"$root/bin/llama-swap" --config "$runtime_config" --config-dir "$root/models.d" --watch-config --listen "0.0.0.0:${PORT:-9292}"
