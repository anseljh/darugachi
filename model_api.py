#!/usr/bin/env python3
"""Small authenticated API for adding safe llama-swap model definitions."""

import argparse
import hmac
import json
import os
import re
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HF_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$")
ENGINES = {"llama-reranker", "llama-embedding", "vllm-pooling"}


def config_for(model_id: str, engine: str, model: str) -> str:
    quoted_id = json.dumps(model_id)
    quoted_model = json.dumps(model)
    if engine == "llama-reranker":
        command = (
            "${env.LLAMA_SERVER} --host 127.0.0.1 --port ${PORT} "
            f"--hf-repo {model} --hf-token ${{env.HF_TOKEN}} --reranking --n-gpu-layers all"
        )
        extra = "    capabilities:\n      reranker: true\n"
    elif engine == "llama-embedding":
        command = (
            "${env.LLAMA_SERVER} --host 127.0.0.1 --port ${PORT} "
            f"--hf-repo {model} --hf-token ${{env.HF_TOKEN}} --embedding --n-gpu-layers all"
        )
        extra = ""
    else:
        container = f"lan-vllm-{model_id}"
        command = (
            f"docker run --init --rm --name {container} --gpus all "
            "-e HF_TOKEN=${env.HF_TOKEN} -v ${env.HF_HOME}:/root/.cache/huggingface "
            f"-p ${{PORT}}:8000 ${{env.VLLM_IMAGE}} --model {model} "
            f"--served-model-name {model_id} --runner pooling"
        )
        extra = f"    cmdStop: docker stop {container}\n"
    return f"models:\n  {quoted_id}:\n{extra}    cmd: {json.dumps(command)}\n"


class Handler(BaseHTTPRequestHandler):
    server: "Server"

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        return hmac.compare_digest(header, f"Bearer {self.server.api_key}")

    def _reply(self, status: HTTPStatus, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != "/models":
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        models = sorted(
            path.stem for path in self.server.models_dir.glob("*.yaml") if not path.name.startswith("_")
        )
        self._reply(HTTPStatus.OK, {"models": models})

    def do_POST(self) -> None:
        if self.path != "/models":
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("body must be 1 to 4096 bytes")
            payload = json.loads(self.rfile.read(size))
            model_id = payload["id"]
            engine = payload["engine"]
            model = payload["model"]
            if not all(isinstance(value, str) for value in (model_id, engine, model)):
                raise ValueError("id, engine, and model must be strings")
            if not MODEL_ID.fullmatch(model_id):
                raise ValueError("invalid id")
            if engine not in ENGINES:
                raise ValueError("invalid engine")
            if not HF_MODEL.fullmatch(model):
                raise ValueError("invalid Hugging Face model")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        path = self.server.models_dir / f"{model_id}.yaml"
        if path.exists():
            self._reply(HTTPStatus.CONFLICT, {"error": "model already exists"})
            return
        with tempfile.NamedTemporaryFile("w", dir=self.server.models_dir, delete=False) as file:
            file.write(config_for(model_id, engine, model))
            temp = Path(file.name)
        temp.replace(path)
        self._reply(HTTPStatus.CREATED, {"id": model_id, "status": "configured"})

    def log_message(self, *_: object) -> None:
        return


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], models_dir: Path, api_key: str):
        super().__init__(address, Handler)
        self.models_dir = models_dir
        self.api_key = api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    api_key = os.environ.get("ADMIN_API_KEY", "")
    if len(api_key) < 32:
        raise SystemExit("ADMIN_API_KEY must be set")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    Server(("0.0.0.0", args.port), args.models_dir, api_key).serve_forever()


if __name__ == "__main__":
    main()
