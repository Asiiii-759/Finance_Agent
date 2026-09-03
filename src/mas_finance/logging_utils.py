from __future__ import annotations

import logging
import sys
import time
from uuid import uuid4

from opentelemetry import trace
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TRACER = trace.get_tracer("mas_finance.api")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


class RequestLoggingMiddleware:
    """Trace HTTP requests without BaseHTTPMiddleware's response-stream bridge."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope["method"])
        path = str(scope["path"])
        request_id = uuid4().hex[:8]
        status_code: int | None = None
        start = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message).append("X-Request-Id", request_id)
            await send(message)

        with _TRACER.start_as_current_span(
            f"{method} {path}",
            attributes={"http.request.method": method, "url.path": path},
        ) as span:
            try:
                await self.app(scope, receive, send_with_request_id)
            finally:
                if status_code is not None:
                    span.set_attribute("http.response.status_code", status_code)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                logging.getLogger("mas_finance.api").info(
                    "request_id=%s method=%s path=%s status=%s duration_ms=%s",
                    request_id,
                    method,
                    path,
                    status_code,
                    elapsed_ms,
                )
