#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin="$root/bin"

if ! grep -qi microsoft /proc/version; then
  echo "Run this inside your WSL2 Linux distribution." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "WSL cannot see the NVIDIA GPU; update the Windows NVIDIA driver first." >&2
  exit 1
fi

if ! command -v curl >/dev/null || ! command -v jq >/dev/null || ! command -v tar >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y curl jq tar
fi

mkdir -p "$bin" "$root/hf-cache"
release=$(curl -fsSL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest)
asset_name=$(jq -r '.assets[] | select(.name | test("^llama-swap_[0-9]+_linux_amd64\\.tar\\.gz$")) | .name' <<<"$release")
asset_url=$(jq -r --arg name "$asset_name" '.assets[] | select(.name == $name) | .browser_download_url' <<<"$release")
checksums_url=$(jq -r '.assets[] | select(.name | endswith("_checksums.txt")) | .browser_download_url' <<<"$release")
if [[ -z "$asset_name" || -z "$asset_url" || -z "$checksums_url" ]]; then
  echo "No Linux amd64 llama-swap release asset found." >&2
  exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL "$asset_url" -o "$tmp/$asset_name"
curl -fL "$checksums_url" -o "$tmp/checksums.txt"
(cd "$tmp" && grep -F " $asset_name" checksums.txt | sha256sum -c -)
tar -xzf "$tmp/$asset_name" -C "$bin" llama-swap
chmod +x "$bin/llama-swap"

[[ -f "$root/.env" ]] || cp "$root/.env.example" "$root/.env"
[[ -f "$root/config.yaml" ]] || cp "$root/config.yaml.example" "$root/config.yaml"

echo "Installed $bin/llama-swap. Edit .env and config.yaml, then run ./run.sh."
