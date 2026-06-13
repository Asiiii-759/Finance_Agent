from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RerankProvider(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        pass


class NoopRerankProvider(RerankProvider):
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        return docs[:top_n]


class DashScopeQwenRerankProvider(RerankProvider):
    def __init__(
        self,
        model: str = "qwen3-rerank",
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1",
        instruct: str | None = None,
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.instruct = instruct
        self.timeout = timeout

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        if not self.api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法调用 qwen3-rerank")

        documents = [doc.get("content", "") for doc in docs]
        body = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(docs)),
        }
        if self.instruct:
            body["instruct"] = self.instruct

        request = urllib.request.Request(
            url=f"{self.base_url}/reranks",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        reranked = []
        for item in payload.get("results", []):
            index = item.get("index")
            if index is None or index >= len(docs):
                continue
            doc = docs[index].copy()
            scores = dict(doc.get("scores", {}))
            scores["rerank"] = item.get("relevance_score")
            doc["scores"] = scores
            doc["rerank_score"] = item.get("relevance_score")
            reranked.append(doc)

        return reranked[:top_n]
