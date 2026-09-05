import json
import os
import signal
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from model_api import (
    ROOT,
    SHUTDOWN_DELAY_SECONDS,
    UPDATE_DELAY_SECONDS,
    Handler,
    Server,
    config_for,
)


class ModelArgumentsTest(unittest.TestCase):
    def test_arguments_are_generic_and_shell_quoted(self):
        arguments = [
            "--hf-overrides",
            '{"architectures":["VoyageQwen3BidirectionalEmbedModel"]}',
            "--pooler-config",
            '{"pooling_type":"MEAN"}',
        ]
        config = config_for(
            "voyage-4-nano",
            "vllm-pooling",
            "voyageai/voyage-4-nano",
            arguments,
        )
        command = json.loads(
            next(line for line in config.splitlines() if line.startswith("    cmd: "))[
                len("    cmd: ") :
            ]
        )

        self.assertIn("--hf-overrides '{\"architectures\":", command)
        self.assertIn("--pooler-config '{\"pooling_type\":\"MEAN\"}'", command)
        generic = config_for("x", "vllm-pooling", "owner/model")
        self.assertNotIn("VoyageQwen3BidirectionalEmbedModel", generic)

    def test_definition_persists_arguments(self):
        arguments = ["--pooling", "mean", "--embd-normalize", "2", "-c", "8192"]
        with tempfile.TemporaryDirectory() as directory:
            handler = Handler.__new__(Handler)
            handler.server = SimpleNamespace(models_dir=Path(directory))
            handler._write_definition(
                "dinghy", "llama-embedding", "owner/model:Q4_K_M", arguments
            )

            metadata = json.loads((Path(directory) / "dinghy.meta.json").read_text())
            self.assertEqual(metadata["arguments"], arguments)
            self.assertIn(
                "--pooling mean --embd-normalize 2 -c 8192",
                (Path(directory) / "dinghy.yaml").read_text(),
            )

    def test_arguments_reject_non_strings(self):
        with self.assertRaisesRegex(ValueError, "arguments"):
            Handler._validate_engine_model(
                "llama-embedding", "owner/model", ["--pooling", 1]
            )


class StartupTimeoutTest(unittest.TestCase):
    def test_api_writes_llama_swap_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = Handler.__new__(Handler)
            handler.path = "/settings"
            handler.server = SimpleNamespace(models_dir=Path(directory))
            handler._log_request = MagicMock()
            handler._authorized = MagicMock(return_value=True)
            handler._read_json_body = MagicMock(
                return_value={"startup_timeout_seconds": 300}
            )
            handler._reply = MagicMock()
            handler.do_PUT()

            self.assertEqual(
                (Path(directory) / "_settings.yaml").read_text(),
                "healthCheckTimeout: 300\n",
            )
            handler._reply.assert_called_once_with(
                HTTPStatus.OK, {"startup_timeout_seconds": 300}
            )

    def test_rejects_invalid_timeout(self):
        handler = Handler.__new__(Handler)
        handler.server = SimpleNamespace(models_dir=Path("unused"))
        for invalid in (True, 14, 3601, "300"):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "startup_timeout_seconds"),
            ):
                handler._write_startup_timeout(invalid)


class SelfUpdateTest(unittest.TestCase):
    @patch("model_api.threading.Timer")
    @patch("model_api.subprocess.Popen")
    def test_update_detaches_then_stops_parent(self, popen, timer_class):
        server = Server.__new__(Server)
        server.update_started = False
        server.update_lock = threading.Lock()
        timer = timer_class.return_value = MagicMock()

        self.assertTrue(server.start_update())
        self.assertFalse(server.start_update())

        command = popen.call_args.args[0]
        self.assertEqual(command[-2:], [str(UPDATE_DELAY_SECONDS), str(ROOT)])
        self.assertIn('git -C "$2" pull --ff-only', command[2])
        self.assertIn('exec "$2/run.sh"', command[2])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        timer_class.assert_called_once_with(
            SHUTDOWN_DELAY_SECONDS, os.kill, (os.getppid(), signal.SIGTERM)
        )
        timer.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
