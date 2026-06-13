import hashlib
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI


class MockEmbeddings:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.dimension = dim

    def _embed_text(self, text: str) -> List[float]:
        vector = []
        seed = text.encode("utf-8", errors="ignore")
        counter = 0
        while len(vector) < self.dim:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
            for byte in digest:
                vector.append((byte / 255.0) - 0.5)
                if len(vector) >= self.dim:
                    break
            counter += 1
        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_dict_list(
        self,
        dict_list: List[Dict[str, Any]],
        content_key: str = "content",
    ) -> List[List[float]]:
        return self.embed_documents([str(doc.get(content_key, "")) for doc in dict_list])


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        dimensions: Optional[int] = None,
        batch_size: int = 10,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.dimension = dimensions
        self.batch_size = max(1, batch_size)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            kwargs = {"model": self.model_name, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**kwargs)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            if embeddings and self.dimension is None:
                self.dimension = len(embeddings[0])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """基础方法：处理单条文本"""
        return self.embed_documents([text])[0]

    def embed_dict_list(
        self, 
        dict_list: List[Dict[str, Any]], 
        content_key: str = "content"
    ) -> List[List[float]]:
        """
        直接传入字典列表，提取文本并向量化，仅返回向量列表。
        """
        if not dict_list:
            return []
        
        texts = [str(doc.get(content_key, "")) for doc in dict_list]
    
        embeddings = self.embed_documents(texts)
        return embeddings


class AliyunBailianEmbeddings(OpenAICompatibleEmbeddings):
    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions: int = 1024,
        batch_size: int = 10,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法调用阿里云百炼 embedding")
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            dimensions=dimensions,
            batch_size=batch_size,
            timeout=timeout,
            max_retries=max_retries,
        )


VllmBgeEmbeddings = OpenAICompatibleEmbeddings
