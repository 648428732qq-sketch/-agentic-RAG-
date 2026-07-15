from __future__ import annotations

from qdrant_client import QdrantClient

import config


def create_qdrant_client() -> QdrantClient:
    """Create the configured local or remote Qdrant client without logging secrets."""
    if config.QDRANT_URL:
        kwargs = {
            "url": config.QDRANT_URL,
            "prefer_grpc": config.QDRANT_PREFER_GRPC,
        }
        if config.QDRANT_API_KEY:
            kwargs["api_key"] = config.QDRANT_API_KEY
        return QdrantClient(**kwargs)
    return QdrantClient(path=config.QDRANT_DB_PATH)
