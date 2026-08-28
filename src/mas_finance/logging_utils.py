from __future__ import annotations

import logging
import sys
import time
from uuid import uuid4

from fastapi import Request
from opentelemetry import trace

_TRACER = trace.get_tracer("mas_finance.api")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


async def request_logging_middleware(request: Request, call_next):
    logger = logging.getLogger("mas_finance.api")
    request_id = uuid4().hex[:8]
    start = time.perf_counter()
    with _TRACER.start_as_current_span(
        f"{request.method} {request.url.path}",
        attributes={"http.request.method": request.method, "url.path": request.url.path},
    ) as span:
        response = await call_next(request)
        span.set_attribute("http.response.status_code", response.status_code)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-Id"] = request_id
    return response
