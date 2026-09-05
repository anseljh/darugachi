# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single WSL2 service (`run.sh`) that runs on a LAN box with an 8 GB GPU and
switches between llama.cpp and vLLM backends on demand. Two processes:

- **llama-swap** (port 9292, from `bin/llama-swap`) — the OpenAI-compatible
  inference gateway (`/v1/...`, web UI at `/ui`). Starts a model on its first
  request, unloads it after `unloadTimeout` idle, and switches backends by
  killing one process and starting another. Configured from
  `.config.runtime.yaml` (generated from `config.yaml` each run) plus every
  `*.yaml` file in `models.d/`.
- **`model_api.py`** (port 9293) — a small stdlib-only HTTP admin API ("the
  controller") that writes/deletes model definitions in `models.d/` and
  manages background weight downloads. This is the file to edit when
  changing the admin API's behavior.

`run.sh` starts both from the same shell after `source .env` (via `set -a`),
so `model_api.py` inherits `HF_TOKEN`, `HF_HOME`, `VLLM_IMAGE`, etc. directly
from `os.environ` — no separate config loading for those.

## Commands

There is no build step, package manager, or test suite — `model_api.py` is
stdlib-only Python. To check a change:

```bash
python3 -m py_compile model_api.py
```

There's no test file to run; verify handler changes by starting the server
against a scratch `models.d/` and curling it directly (see git history for
`model_api.py` for examples of this pattern — auth checks, create/delete/
download/status transitions, and error paths were all exercised this way
with fake `hf`/`docker` binaries prepended to `PATH` so no real network or
Docker calls are needed).

Setup and running on the actual GPU box:

```bash
./install.sh   # first time: installs bin/llama-swap, generates .env keys
./run.sh       # starts model_api.py (9293) and llama-swap (9292)
```

## Architecture: how a model definition becomes a running process

1. `POST /models` on the admin API (`model_api.py`) validates `id`/`engine`/
   `model` and writes `models.d/{id}.yaml` — a `config_for()`-generated
   llama-swap model block whose `cmd` template still contains unresolved
   `${env.X}` / `${PORT}` placeholders. It also writes a `{id}.meta.json`
   sidecar (engine + model + additional command arguments) alongside, used by
   the download endpoints.
2. llama-swap polls `models.d/` (2s interval) and reloads automatically —
   `model_api.py` never talks to llama-swap directly.
3. The actual model process (llama-server or a `docker run` of vLLM) only
   starts on a client's first `/v1` request to that model id; llama-swap
   resolves the `${env...}` placeholders and downloads Hugging Face weights
   itself via `--hf-repo` (llama.cpp) or the mounted `HF_HOME` cache
   (vLLM) at that point.
4. This lazy-start path has a llama-swap health-check timeout that a large
   cold download or slow model initialization can exceed. `POST
   /models/{id}/download` exists to sidestep this — it's the end-to-end
   entry point: if `{id}` isn't registered yet, the request body's
   `engine`/`model` register it first (same write path as `POST /models`,
   factored into `Handler._write_definition`); if it's already registered,
   the body can be omitted and its existing `{id}.meta.json` is reused.
   Either way it then runs `hf download` (scoped to the GGUF quant tag via
   `--include` for the two llama engines) or `docker pull` + `hf download`
   (for `vllm-pooling`) via `Server._run_tracked`, in a background thread
   inside `model_api.py` itself, independent of llama-swap. `_run_tracked`
   iterates the subprocess's merged stdout/stderr line by line — Python's
   universal-newline text mode splits on a bare `\r` too, so a tqdm-style
   progress bar that rewrites one line still yields one "line" per update —
   and stores the latest line plus a timestamp in `self.downloads[model_id]`
   after every update. `GET /models/{id}/status` reports `not_started` /
   `downloading` / `ready` / `failed`, and while `downloading` also reports
   that latest line as `message` plus a `seconds_since_update` computed at
   request time, so a caller can tell a stalled download from a merely slow
   one without needing to correlate timestamps itself. Poll until `ready`,
   then send the inference request. `POST /models` still exists on its own
   for registering a definition without immediately downloading it.
   `PUT /settings` accepts `startup_timeout_seconds` (15--3600) and persists
   llama-swap's process-wide `healthCheckTimeout` in the watched
   `models.d/_settings.yaml` fragment, so slow vLLM startups can be allowed
   without model-name branches or another restart.
5. `DELETE /models/{id}` removes the yaml and meta sidecar and clears
   in-memory download state, but does not stop a currently running instance
   or delete cached weights — cache/process cleanup is manual.

Download status is tracked only in an in-memory dict on the `Server`
instance; it resets on restart and isn't visible across `model_api.py`
processes.

`GET /spec` serves `openapi.yaml` from disk as-is and is the one admin-API
endpoint that skips bearer auth (it's static documentation, not data).

## The three engines

`config_for()` in `model_api.py` is the single place that knows how to turn
`(engine, model)` into a llama-swap `cmd`:

- `llama-reranker` / `llama-embedding` — both shell out to `${env.LLAMA_SERVER}`
  with `--hf-repo {model}`. `model` is a Hugging Face repo id, optionally
  suffixed `:QUANT` (e.g. `Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M`) — this only
  works for repos that actually publish GGUF files; a safetensors-only repo
  (like the base, non-GGUF Qwen models) will fail at load time, not at
  registration time, since `model_api.py` doesn't fetch or inspect the repo.
- `vllm-pooling` — runs `docker run ... ${env.VLLM_IMAGE} --model {model}
  --runner pooling`, for safetensors models vLLM can serve directly (no GGUF
  needed). Requires `VLLM_IMAGE` to be set in `.env`; if it's blank,
  llama-swap's config reload fails outright for every model in the same
  reload pass, not just the vLLM one.

## Known environment gotcha

`llama-server` must be built with `-DLLAMA_OPENSSL=ON` (default, but only if
`libssl-dev` is installed before running cmake). Without it, `--hf-repo`
fails at model-load time with `get_repo_commit: error: HTTPS is not
supported`, and llama-swap reports it upstream only as `upstream command
exited prematurely` — the real cause is only visible in `llama-server`'s own
stderr, not in llama-swap's `/logs`, which merges request/lifecycle logging
but not each child process's own output.

## Line endings

`.gitattributes` pins `.sh`/`.py`/`.yaml`/`.env.example` to LF. This box runs
these files directly from WSL; a checkout on a mounted Windows drive
(`/mnt/*`) can otherwise introduce CRLF and break `#!` shebangs.
