from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

import config
from db.qdrant_client_factory import create_qdrant_client
from db.vector_db_manager import _embedding_model_kwargs


class RuntimePortabilityTests(unittest.TestCase):
    def test_env_float_parses_value_and_falls_back(self) -> None:
        with patch.dict("os.environ", {"TCM_TEST_FLOAT": "60.5"}):
            self.assertEqual(config._env_float("TCM_TEST_FLOAT", 12.0), 60.5)

        with patch.dict("os.environ", {"TCM_TEST_FLOAT": "invalid"}):
            self.assertEqual(config._env_float("TCM_TEST_FLOAT", 12.0), 12.0)

    def test_local_qdrant_uses_configured_path(self) -> None:
        with (
            patch.object(config, "QDRANT_URL", ""),
            patch.object(config, "QDRANT_DB_PATH", "/tmp/tcm-qdrant"),
            patch("db.qdrant_client_factory.QdrantClient") as client,
        ):
            create_qdrant_client()

        client.assert_called_once_with(path="/tmp/tcm-qdrant")

    def test_remote_qdrant_uses_url_without_local_path(self) -> None:
        with (
            patch.object(config, "QDRANT_URL", "http://127.0.0.1:6333"),
            patch.object(config, "QDRANT_API_KEY", "test-secret"),
            patch.object(config, "QDRANT_PREFER_GRPC", True),
            patch("db.qdrant_client_factory.QdrantClient") as client,
        ):
            create_qdrant_client()

        client.assert_called_once_with(
            url="http://127.0.0.1:6333",
            api_key="test-secret",
            prefer_grpc=True,
        )

    def test_embedding_auto_device_is_left_to_sentence_transformers(self) -> None:
        with (
            patch.object(config, "EMBEDDING_DEVICE", "auto"),
            patch.object(config, "EMBEDDING_LOCAL_FILES_ONLY", True),
        ):
            kwargs = _embedding_model_kwargs()

        self.assertEqual(kwargs, {"local_files_only": True})

    def test_embedding_cuda_device_is_explicit(self) -> None:
        with patch.object(config, "EMBEDDING_DEVICE", "cuda"):
            kwargs = _embedding_model_kwargs()

        self.assertEqual(kwargs["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
