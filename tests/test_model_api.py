import unittest

from model_api import config_for


class ConfigForTest(unittest.TestCase):
    def test_voyage_nano_gets_its_embedding_contract(self):
        config = config_for(
            "voyage-4-nano", "vllm-pooling", "voyageai/voyage-4-nano"
        )

        for argument in (
            "--convert embed",
            "VoyageQwen3BidirectionalEmbedModel",
            "--pooler-config",
            "--dtype bfloat16",
            "--enforce-eager",
        ):
            self.assertIn(argument, config)

        generic = config_for("other", "vllm-pooling", "owner/other")
        self.assertNotIn("VoyageQwen3BidirectionalEmbedModel", generic)


if __name__ == "__main__":
    unittest.main()
