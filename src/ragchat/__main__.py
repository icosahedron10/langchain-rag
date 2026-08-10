"""Run the ragchat API with Uvicorn."""

from __future__ import annotations

import uvicorn

from ragchat.app import create_app
from ragchat.config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
