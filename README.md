# Darugachi

![A Mongol rider with an administrator, public domain](docs/500px-Mongol_Rider_with_Administrator.jpeg)

*[Mongol Rider with Administrator](https://commons.wikimedia.org/wiki/File:Mongol_Rider_with_Administrator.jpg)*

A *darugachi* was the official a Mongol khan installed in a conquered province to collect tribute on his behalf. This project is the same idea for a GPU, such as one in a gaming PC a parent bought for a teenager: a WSL2 service that switches a GPU between llama.cpp and vLLM, collecting inference work from whoever controls the box. Requests use the normal OpenAI-compatible endpoint at `http://LAN-BOX:9292/v1`; the web UI is `/ui`. A model starts on its first request and unloads after five idle minutes. `POST /api/models/unload` releases everything immediately.

## Install and start

Inside an Ubuntu WSL2 shell on the tributary machine with the GPU:

```bash
git clone https://github.com/anseljh/darugachi.git
cd darugachi
./install.sh
# edit .env and config.yaml; API_KEY was generated automatically
./run.sh
```

If an existing checkout on a mounted Windows drive reports `bash\r: No such
file or directory` or `.env: ... $'\r': command not found`, normalize its
scripts and generated configuration once, then retry:

```bash
sed -i 's/\r$//' install.sh run.sh .env config.yaml
./install.sh
```

`.gitattributes` pins scripts and generated-config templates to LF so fresh
clones do not have this problem.

## Manual setup

1. Install a current Windows NVIDIA driver. In WSL, `nvidia-smi` must work. Do **not** install a Linux NVIDIA driver.
2. Install [Docker Desktop with WSL integration](https://docs.docker.com/desktop/features/wsl/) (or Docker Engine in WSL). `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` must work. Add your Linux user to the `docker` group if needed.
3. Build or install [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) in WSL, then set its absolute Linux path in `.env`. The existing LAN llama.cpp build is fine.
4. Create a read-only Hugging Face token and set `HF_TOKEN`. `install.sh` generates and saves long random `API_KEY` and `ADMIN_API_KEY` values in `.env`. `LLAMA_HF_REPO` enables the default GGUF reranker; `VLLM_MODEL` and `VLLM_IMAGE` together enable the default vLLM pooling model. Leave any of them blank to skip that default model and add models remotely instead.
5. Make port 9292 reachable from the dev machine: use WSL mirrored networking or forward the port through Windows, and restrict Windows Firewall to the dev machine. Never expose it broadly; the API key protects requests but the LAN boundary should too.

### Build `llama-server` with CUDA

Build in the WSL Linux filesystem:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git nvidia-cuda-toolkit libssl-dev
mkdir -p "$HOME/src"
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$HOME/src/llama.cpp"
cmake -S "$HOME/src/llama.cpp" -B "$HOME/src/llama.cpp/build" \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/src/llama.cpp/build" --target llama-server --config Release -j 2
"$HOME/src/llama.cpp/build/bin/llama-server" --list-devices
```

The last command must list the NVIDIA GPU. Without `libssl-dev` present before
this build, cmake silently builds `llama-server` without TLS support: it
starts fine but `--hf-repo` fails at model-load time with `get_repo_commit:
error: HTTPS is not supported`. If you hit that, install `libssl-dev` and
rebuild.

Then set this in `.env` (replace the example value):

```bash
LLAMA_SERVER=$HOME/src/llama.cpp/build/bin/llama-server
```

The controller uses `--hf-repo` to download GGUF models, which current
`llama-server` supports. If `nvcc` is still unavailable after the package
install, stop there and capture the output of `lsb_release -a` and
`nvidia-smi` rather than installing a Linux NVIDIA driver.

### Reach the controller from the LAN

Keep `llama-server` private to WSL; expose only the controller's inference
port (9292) and, when needed, its administration port (9293). First verify it
locally in WSL:

```bash
curl -i http://127.0.0.1:9292/
```

Then open an **Administrator PowerShell** on the Windows LAN box. Replace
`DEV_MACHINE_IP` with the dev machine's LAN IP address:

```powershell
$wslIp = (wsl.exe hostname -I).Trim().Split(' ')[0]

netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=9292 connectaddress=$wslIp connectport=9292
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=9293 connectaddress=$wslIp connectport=9293

New-NetFirewallRule -DisplayName "Darugachi inference" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9292 -RemoteAddress DEV_MACHINE_IP -Profile Private
New-NetFirewallRule -DisplayName "Darugachi admin" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9293 -RemoteAddress DEV_MACHINE_IP -Profile Private
```

Port 9293 can add models: never allow it from `Any` or `LocalSubnet`. WSL2's
NAT address can change after a restart; rerun the two `netsh` commands if the
controller becomes unreachable.

For a batch run that must unload immediately rather than wait for the TTL:

```bash
curl -X POST -H "Authorization: Bearer $API_KEY" http://LAN-BOX:9292/api/models/unload
```

vLLM pooling supports embedding and rerank APIs when the selected model supports them. The default vLLM model is registered only when both `VLLM_MODEL` and `VLLM_IMAGE` are set; otherwise add a model remotely.

Nomolith's evaluated 8 GB GPU model shortlist and copy-paste `add` commands
are in [MODELS.md](MODELS.md).

## Add a model remotely

`run.sh` starts a small management API on port 9293. It writes only validated model definitions to `models.d/`; llama-swap watches that directory and makes a new model available automatically. Its first inference request downloads the Hugging Face model if missing.

Every request to this API logs one line to stdout: a UTC timestamp, the HTTP
method, the endpoint (e.g. `/models/{id}/download`), and `model=ID` (or `-`
when the request has no associated model, or its id couldn't be determined).

```bash
export ADMIN_API_KEY='value from the LAN box .env file'
curl -X POST http://LAN-BOX:9293/models \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"id":"bge-reranker","engine":"llama-reranker","model":"Geofront/BGE-Reranker-v2-M3-GGUF:Q4_K_M","arguments":["--ctx-size","8192"]}'
```

`engine` is one of `llama-reranker`, `llama-embedding`, or `vllm-pooling`; `model` is its Hugging Face ID (with optional GGUF quant after `:`). `GET /models` lists definitions created by this API. `DELETE /models/ID` removes a definition's yaml file (llama-swap picks up the removal automatically; it does not stop a currently running instance or delete cached weights). Port 9293 is an admin interface: firewall it to the dev machine and do not use the regular inference `API_KEY` there.

`arguments` is an optional array of additional command arguments. The API
shell-quotes each item, writes it into the llama-swap definition, and persists
the original array in the model's metadata sidecar for later downloads. Model
flags belong here, not in controller code.

llama-swap's startup health timeout is process-wide. Raise it through the
admin API when a slow vLLM model needs more than the 120-second default; the
watched setting is persisted in `models.d/_settings.yaml` and applies without
another restart:

```bash
curl -X PUT http://LAN-BOX:9293/settings \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"startup_timeout_seconds":300}'
```

A first inference request downloads the model lazily, but llama-swap's startup
health check can expire during a large cold download or slow initialization.
`POST
/models/ID/download` downloads end-to-end instead, registering `ID` first if
it doesn't already exist:

```bash
curl -X POST http://LAN-BOX:9293/models/bge-reranker/download \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"engine":"llama-reranker","model":"Geofront/BGE-Reranker-v2-M3-GGUF:Q4_K_M"}'
curl http://LAN-BOX:9293/models/bge-reranker/status -H "Authorization: Bearer $ADMIN_API_KEY"
```

The body (`engine`, `model`, and optional `arguments`, same shape as `POST
/models`) is only required
the first time for a given `ID`; once it's registered, retriggering a download
(e.g. after a `failed` status) needs no body — its existing
`engine`/`model`/`arguments` are reused. Either way this runs `hf download` (for `llama-reranker`/
`llama-embedding`, scoped to the GGUF quant via `--include`) or `docker pull`
+ `hf download` (for `vllm-pooling`) in the background and returns
immediately. `GET /models/ID/status` reports `not_started`, `downloading`,
`ready`, or `failed` (with a `message` on failure). While `downloading`, it
also reports `message` (the underlying command's latest progress line —
percentage and transfer rate, as printed), `updated_at` (when that line was
seen), and `seconds_since_update` (computed at request time — a large or
growing value means the download has stalled, not just that it's slow). Poll
status until `ready`, then send an inference request as normal.

`GET /spec` serves this API's [OpenAPI spec](openapi.yaml) as-is; unlike the
other endpoints it needs no `Authorization` header.

## Update remotely

The authenticated update endpoint hands restart responsibility to a detached
process, then stops the running controller and inference gateway. After three
seconds the detached process pulls a fast-forward update and starts Darugachi
again. If the pull fails, it restarts the current checkout. Output is appended
to `/tmp/darugachi-update.log`.

```bash
curl -X POST http://LAN-BOX:9293/self-update \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```
