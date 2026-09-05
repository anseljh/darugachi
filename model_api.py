#!/usr/bin/env python3
"""Small authenticated API for adding safe llama-swap model definitions."""

import argparse
import hmac
import json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HF_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$")
ENGINES = {"llama-reranker", "llama-embedding", "vllm-pooling"}
SPEC_PATH = Path(__file__).parent / "openapi.yaml"
ROOT = Path(__file__).resolve().parent
UPDATE_DELAY_SECONDS = 3
SHUTDOWN_DELAY_SECONDS = 1


def config_for(
    model_id: str, engine: str, model: str, arguments: list[str] | None = None
) -> str:
    quoted_id = json.dumps(model_id)
    extra_arguments = f" {shlex.join(arguments)}" if arguments else ""
    if engine == "llama-reranker":
        command = (
            "${env.LLAMA_SERVER} --host 127.0.0.1 --port ${PORT} "
            f"--hf-repo {model} --hf-token ${{env.HF_TOKEN}} "
            f"--embedding --pooling rank --reranking --n-gpu-layers all{extra_arguments}"
        )
        extra = "    capabilities:\n      reranker: true\n"
    elif engine == "llama-embedding":
        command = (
            "${env.LLAMA_SERVER} --host 127.0.0.1 --port ${PORT} "
            f"--hf-repo {model} --hf-token ${{env.HF_TOKEN}} --embedding "
            f"--n-gpu-layers all{extra_arguments}"
        )
        extra = ""
    else:
        container = f"lan-vllm-{model_id}"
        command = (
            f"docker run --init --rm --name {container} --gpus all "
            "-e HF_TOKEN=${env.HF_TOKEN} -v ${env.HF_HOME}:/root/.cache/huggingface "
            f"-p ${{PORT}}:8000 ${{env.VLLM_IMAGE}} --model {model} "
            f"--served-model-name {model_id} --runner pooling "
            # vLLM reserves this fraction of the whole CARD up front, before it
            # loads any weights, and aborts if that much is not already free.
            # The 0.92 default assumes a dedicated GPU. This box has ~1.4GiB held
            # by WSL2/display, so 0.92 of 8GiB (7.36) exceeded the 6.56 free and
            # a 596MB embedding model failed to start -- after a 45s torch import,
            # which reads as a model problem rather than an arithmetic one.
            # A property of the card and what else is resident, not of the model.
            f"--gpu-memory-utilization ${{env.VLLM_GPU_MEMORY_UTILIZATION}} "
            # Most modern pooling models on the Hub declare their config through
            # an auto_map on a model_type transformers does not know, so the
            # config will not load without this even when vLLM implements the
            # architecture itself and imports nothing from the repo. Serving a
            # repo you deliberately named is the trust boundary this controller
            # already has -- the admin API is bearer-authed and firewalled.
            "--trust-remote-code "
            # Without this vLLM sizes the KV cache from the config's position
            # embeddings, which is 131072 for some 1B models -- far past what an
            # 8GB card can hold, and past what the model card says the model
            # actually supports. The cap belongs to the box, not the model.
            f"--max-model-len ${{env.VLLM_MAX_MODEL_LEN}}{extra_arguments}"
        )
        extra = f"    cmdStop: docker stop {container}\n"
    return f"models:\n  {quoted_id}:\n{extra}    cmd: {json.dumps(command)}\n"


class Handler(BaseHTTPRequestHandler):
    server: "Server"

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        return hmac.compare_digest(header, f"Bearer {self.server.api_key}")

    def _log_request(self, endpoint: str, model_id: str | None) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"{timestamp} {self.command} {endpoint} model={model_id or '-'}", flush=True)

    def _reply(self, status: HTTPStatus, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 4096:
            raise ValueError("body must be 1 to 4096 bytes")
        return json.loads(self.rfile.read(size))

    @staticmethod
    def _validate_engine_model(
        engine: object, model: object, arguments: object
    ) -> None:
        if not isinstance(engine, str) or not isinstance(model, str):
            raise ValueError("engine and model must be strings")
        if engine not in ENGINES:
            raise ValueError("invalid engine")
        if not HF_MODEL.fullmatch(model):
            raise ValueError("invalid Hugging Face model")

        if not isinstance(arguments, list) or len(arguments) > 64:
            raise ValueError("arguments must be an array of at most 64 strings")
        if any(
            not isinstance(argument, str)
            or len(argument) > 512
            or "\0" in argument
            for argument in arguments
        ):
            raise ValueError("arguments must contain strings of at most 512 characters")

    def _write_definition(
        self, model_id: str, engine: str, model: str, arguments: list[str]
    ) -> None:
        path = self.server.models_dir / f"{model_id}.yaml"
        with tempfile.NamedTemporaryFile("w", dir=self.server.models_dir, delete=False) as file:
            file.write(config_for(model_id, engine, model, arguments))
            temp = Path(file.name)
        temp.replace(path)
        meta_path = self.server.models_dir / f"{model_id}.meta.json"
        meta_path.write_text(
            json.dumps({"engine": engine, "model": model, "arguments": arguments})
        )

    def do_GET(self) -> None:
        if self.path == "/spec":
            self._log_request("/spec", None)
            try:
                data = SPEC_PATH.read_bytes()
            except OSError:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "spec not found"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/yaml")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/models/") and self.path.endswith("/status"):
            self._download_status()
            return
        if self.path != "/models":
            self._log_request(self.path, None)
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._log_request("/models", None)
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        models = sorted(
            path.stem for path in self.server.models_dir.glob("*.yaml") if not path.name.startswith("_")
        )
        self._reply(HTTPStatus.OK, {"models": models})

    def _download_status(self) -> None:
        model_id = self.path[len("/models/"):-len("/status")]
        self._log_request("/models/{id}/status", model_id)
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if not MODEL_ID.fullmatch(model_id):
            self._reply(HTTPStatus.BAD_REQUEST, {"error": "invalid id"})
            return
        if not (self.server.models_dir / f"{model_id}.yaml").exists():
            self._reply(HTTPStatus.NOT_FOUND, {"error": "model not found"})
            return
        status = dict(self.server.downloads.get(model_id, {"state": "not_started"}))
        updated_at = status.get("updated_at")
        if updated_at is not None:
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)
            status["seconds_since_update"] = round(elapsed.total_seconds())
        self._reply(HTTPStatus.OK, {"id": model_id, **status})

    def do_POST(self) -> None:
        if self.path == "/self-update":
            self._start_update()
            return
        if self.path.startswith("/models/") and self.path.endswith("/download"):
            self._start_download()
            return
        if self.path != "/models":
            self._log_request(self.path, None)
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorized():
            self._log_request("/models", None)
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        model_id = None
        try:
            payload = self._read_json_body()
            model_id = payload["id"]
            engine = payload["engine"]
            model = payload["model"]
            arguments = payload.get("arguments", [])
            if not isinstance(model_id, str):
                raise ValueError("id must be a string")
            if not MODEL_ID.fullmatch(model_id):
                raise ValueError("invalid id")
            self._validate_engine_model(engine, model, arguments)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._log_request("/models", model_id if isinstance(model_id, str) else None)
            self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._log_request("/models", model_id)

        path = self.server.models_dir / f"{model_id}.yaml"
        if path.exists():
            self._reply(HTTPStatus.CONFLICT, {"error": "model already exists"})
            return
        self._write_definition(model_id, engine, model, arguments)
        self._reply(HTTPStatus.CREATED, {"id": model_id, "status": "configured"})

    def _start_update(self) -> None:
        self._log_request("/self-update", None)
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            scheduled = self.server.start_update()
        except OSError:
            self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "could not schedule update"})
            return
        if not scheduled:
            self._reply(HTTPStatus.CONFLICT, {"error": "update already scheduled"})
            return
        self._reply(
            HTTPStatus.ACCEPTED,
            {"status": "updating", "restart_in_seconds": UPDATE_DELAY_SECONDS},
        )

    def _start_download(self) -> None:
        model_id = self.path[len("/models/"):-len("/download")]
        self._log_request("/models/{id}/download", model_id)
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if not MODEL_ID.fullmatch(model_id):
            self._reply(HTTPStatus.BAD_REQUEST, {"error": "invalid id"})
            return
        if self.server.downloads.get(model_id, {}).get("state") == "downloading":
            self._reply(HTTPStatus.CONFLICT, {"error": "download already in progress"})
            return

        yaml_path = self.server.models_dir / f"{model_id}.yaml"
        if yaml_path.exists():
            meta = json.loads((self.server.models_dir / f"{model_id}.meta.json").read_text())
            engine, model = meta["engine"], meta["model"]
        else:
            try:
                payload = self._read_json_body()
                engine = payload["engine"]
                model = payload["model"]
                arguments = payload.get("arguments", [])
                self._validate_engine_model(engine, model, arguments)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._write_definition(model_id, engine, model, arguments)

        self.server.downloads[model_id] = {"state": "downloading"}
        threading.Thread(
            target=self.server.run_download,
            args=(model_id, engine, model),
            daemon=True,
        ).start()
        self._reply(HTTPStatus.ACCEPTED, {"id": model_id, "status": "downloading"})

    def do_DELETE(self) -> None:
        prefix = "/models/"
        if not self.path.startswith(prefix):
            self._log_request(self.path, None)
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        model_id = self.path[len(prefix):]
        self._log_request("/models/{id}", model_id)
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if not MODEL_ID.fullmatch(model_id):
            self._reply(HTTPStatus.BAD_REQUEST, {"error": "invalid id"})
            return
        path = self.server.models_dir / f"{model_id}.yaml"
        if not path.exists():
            self._reply(HTTPStatus.NOT_FOUND, {"error": "model not found"})
            return
        path.unlink()
        (self.server.models_dir / f"{model_id}.meta.json").unlink(missing_ok=True)
        self.server.downloads.pop(model_id, None)
        self._reply(HTTPStatus.OK, {"id": model_id, "status": "deleted"})

    def log_message(self, *_: object) -> None:
        # Suppresses BaseHTTPRequestHandler's default access-log line; _log_request logs instead.
        return


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], models_dir: Path, api_key: str):
        super().__init__(address, Handler)
        self.models_dir = models_dir
        self.api_key = api_key
        self.downloads: dict[str, dict] = {}
        self.update_started = False
        self.update_lock = threading.Lock()

    def start_update(self) -> bool:
        with self.update_lock:
            if self.update_started:
                return False
            self.update_started = True
        try:
            log = open("/tmp/darugachi-update.log", "ab", buffering=0)
            subprocess.Popen(
                [
                    "sh",
                    "-c",
                    'sleep "$1"; git -C "$2" pull --ff-only; exec "$2/run.sh"',
                    "darugachi-update",
                    str(UPDATE_DELAY_SECONDS),
                    str(ROOT),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            with self.update_lock:
                self.update_started = False
            raise
        finally:
            if "log" in locals():
                log.close()
        timer = threading.Timer(
            SHUTDOWN_DELAY_SECONDS, os.kill, (os.getppid(), signal.SIGTERM)
        )
        timer.daemon = True
        timer.start()
        return True

    def run_download(self, model_id: str, engine: str, model: str) -> None:
        repo, _, tag = model.partition(":")
        try:
            if engine == "vllm-pooling":
                image = os.environ.get("VLLM_IMAGE", "")
                if image:
                    self._run_tracked(model_id, ["docker", "pull", image], "pulling image")
                self._run_tracked(model_id, ["hf", "download", repo], "downloading weights")
            else:
                command = ["hf", "download", repo]
                if tag:
                    command += ["--include", f"*{tag}*"]
                self._run_tracked(model_id, command, "downloading weights")
            self.downloads[model_id] = {"state": "ready"}
        except subprocess.CalledProcessError as error:
            self.downloads[model_id] = {"state": "failed", "message": (error.stderr or "")[-2000:]}
        except OSError as error:
            self.downloads[model_id] = {"state": "failed", "message": str(error)}

    def _run_tracked(self, model_id: str, command: list[str], phase: str) -> None:
        # hf/docker progress bars rewrite their line with \r; Python's universal-newline
        # text mode still splits on that, so iterating the pipe yields one line per update.
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        last_line = ""
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            last_line = line
            self.downloads[model_id] = {
                "state": "downloading",
                "phase": phase,
                "message": last_line,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command, output=last_line, stderr=last_line)


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
