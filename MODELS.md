# Nomolith-evaluated models

These model definitions are suitable for the LAN box's 8 GB RTX 3050. Add
several definitions, but load only one model at a time.

Set the controller administration key on the dev machine:

```bash
export LAN_MODEL_ADMIN_API_KEY='admin key from the LAN box .env'
```

## Add first

These rerankers were evaluated through Nomolith's prior llama.cpp LAN route on
this GPU. `zerank-1` has the best measured result; `zerank-1-small` is the
faster, smaller option.

| ID | Nomolith nDCG@10 | Median run time |
| --- | ---: | ---: |
| `zerank-1-small` | 0.740 | 17.9 s |
| `zerank-1` | 0.763 | 47.0 s |
| `zerank-2` | 0.760 | 29.6 s |

```bash
uv run python scripts/lan_model_controller.py add \
  zerank-1-small llama-reranker seamon67/Zerank-1-Small-GGUF:Q8_0
uv run python scripts/lan_model_controller.py add \
  zerank-1 llama-reranker seamon67/Zerank-1-GGUF:Q8_0
uv run python scripts/lan_model_controller.py add \
  zerank-2 llama-reranker seamon67/Zerank-2-GGUF:Q8_0

uv run python scripts/lan_model_controller.py add \
  qwen3-embed-4b llama-embedding Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M
```

`qwen3-embed-4b` was evaluated in Nomolith's hosted embedding run. Its Q4
GGUF is about 2.5 GB, so it is the local embedding starting point. See the
[official model files](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/tree/main).

## Add after the first smoke test

```bash
uv run python scripts/lan_model_controller.py add \
  qwen3-embed-8b llama-embedding Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M
```

`qwen3-embed-8b` was Nomolith's highest-scoring evaluated embedding model
(nDCG@10 0.877). Its Q4 GGUF is about 4.68 GB, so it is plausible but
borderline on an 8 GB card; load it alone and verify an embedding request.
See the [official model files](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF).

The previously evaluated `giladgd/Qwen3-Reranker-4B-GGUF:Q8_0` and
`giladgd/Qwen3-Reranker-8B-GGUF:Q5_K_M` were also served on the LAN box, but
the Zerank models above are the better starting set.

## Not supported by the generic controller yet

- `nvidia/Nemotron-3-Embed-1B-BF16` scored 0.849, but correct use requires
  separate `query` and `passage` input prefixes and NVIDIA's specified vLLM
  version.
- `nvidia/llama-nemotron-rerank-1b-v2` needs NVIDIA's score template -- a Jinja
  template producing `question:... passage:...`, without which the scores are
  simply wrong. `vllm-pooling` supplies no template and serves `/v1/embeddings`
  rather than `/rerank`, so this needs an engine of its own. Its
  `--trust-remote-code` requirement is now met; the template is not. See its
  [model card](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2).

## Newly servable

- `voyageai/voyage-4-nano`. Its `vllm-pooling` definition selects vLLM's
  native bidirectional Voyage architecture, mean pooling, BF16, and the embed
  conversion path. The endpoint returns the finished 2048-dimensional vector;
  clients still supply the query/document prompt.
- `nvidia/llama-nemotron-embed-vl-1b-v2` (embedding, not the reranker above).
  vLLM implements `LlamaNemotronVLModel` natively as an embedding model from
  0.17.0 onward, so `vllm-pooling` can serve it now that the definition passes
  `--trust-remote-code` and `--max-model-len`. 1.68B params, 3.36GB of BF16.
  The endpoint applies no prompt prefixes on the plain `input` route, so a
  client must send `query: ` / `passage: ` itself.

After adding a definition, the first inference downloads it. Run
`uv run python scripts/lan_model_controller.py list` to inspect definitions;
run `uv run python scripts/lan_model_controller.py unload ID` after an
evaluation to release the GPU.
