import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_api import Handler, config_for


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


if __name__ == "__main__":
    unittest.main()
