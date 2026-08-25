# LAN model server

One WSL2 service that switches the 8 GB GPU between llama.cpp and vLLM. Requests use the normal OpenAI-compatible endpoint at `http://LAN-BOX:9292/v1`; the web UI is `/ui`. A model starts on its first request and unloads after five idle minutes. `POST /api/models/unload` releases everything immediately.

## Install and start

Inside an Ubuntu WSL2 shell on the LAN box:

```bash
git clone https://github.com/anseljh/lan-model-server.git
cd lan-model-server
./install.sh
# edit .env and config.yaml
./run.sh
```

## Manual setup

1. Install a current Windows NVIDIA driver. In WSL, `nvidia-smi` must work. Do **not** install a Linux NVIDIA driver.
2. Install Docker Desktop with WSL integration (or Docker Engine in WSL). `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` must work. Add your Linux user to the `docker` group if needed.
3. Build or install `llama-server` in WSL, then set its absolute Linux path in `.env`. The existing LAN llama.cpp build is fine.
4. Create a read-only Hugging Face token and set `HF_TOKEN`; set a long random `API_KEY`. Choose `LLAMA_HF_REPO` (a GGUF repository and quant) and `VLLM_MODEL` (a Safetensors pooling model). Both engines fetch to their persistent cache on first use.
5. Make port 9292 reachable from the dev machine: use WSL mirrored networking or forward the port through Windows, and restrict Windows Firewall to the dev machine. Never expose it broadly; the API key protects requests but the LAN boundary should too.

For a batch run that must unload immediately rather than wait for the TTL:

```bash
curl -X POST -H "Authorization: Bearer $API_KEY" http://LAN-BOX:9292/api/models/unload
```

vLLM pooling supports embedding and rerank APIs when the selected model supports them. Keep its model-specific flags in `config.yaml`; no new controller code is needed.
