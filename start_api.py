from __future__ import annotations

import uvicorn

from mas_finance.api.app import create_app
from mas_finance.config import AppConfig


def main() -> None:
    config = AppConfig.from_env()
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
