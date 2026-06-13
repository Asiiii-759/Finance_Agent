import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from Finance_RAG.settings import Settings
from Finance_RAG.utils import build_logger
from Finance_RAG.parser_chunk_search.embedding import (
    AliyunBailianEmbeddings,
    MockEmbeddings,
    OpenAICompatibleEmbeddings,
)
from Finance_RAG.parser_chunk_search.native_faiss import NativeFAISS

from Finance_RAG.db.knowledge_repository import (
    # 知识库相关 (KB)
    kb_exists,
    add_kb_to_db,
    # 实验配置相关 (Experiment)
    experiment_exists,
    list_experiments_from_db,
    delete_experiment_from_db,
    add_experiment_to_db,
    # 文件管理相关 (File)
    file_exists_in_kb,
    add_file_to_db,
    delete_file_from_db,
    # 文本块/切片相关 (Chunk)
    add_chunk_to_db_by_expName,
    delete_chunk_from_db_by_expName_fileName,
    list_chunkId_from_db_by_expName_fileName,
)

from .chunker import (
    KnowledgeFile,
    get_vs_path,
    get_kb_path,
    get_doc_path,
    get_json_path
)

logger = build_logger()


def tokenize_for_search(text: str) -> List[str]:
    return jieba.lcut_for_search(text or "")


def top_score_indices(scores, limit: int) -> List[int]:
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:limit]


class SupportedVSType:
    FAISS = "faiss"
    MILVUS = "milvus"
    

class KBService(ABC):
    def __init__(
        self,
        knowledge_base_name: str = None,
        exp_name: str = None,
        kb_info: str = None,
        index_type: str = Settings.kb_settings.INDEX_TYPE,
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE
    ):
        self.kb_name = knowledge_base_name
        self.exp_name = exp_name
        if not self.exp_name:
            raise ValueError("未指定实验名称！")
        
        self.kb_info = kb_info or f"关于{knowledge_base_name}的知识库"
        self.embed_model = embed_model
        
        self.kb_path = get_kb_path(self.kb_name)
        self.file_path = get_doc_path(self.kb_name)
        self.json_path = get_json_path(self.kb_name)
        self.create_kb()
        if self.exp_name:
            self.vs_path = get_vs_path(self.kb_name, self.exp_name)
            os.makedirs(self.vs_path, exist_ok=True)
            self.create_experiment(chunk_size, index_type, chunk_overlap)
            self.do_init()

    def __repr__(self) -> str:
        return f"KB: {self.kb_name} | Exp: {self.exp_name} @ {self.embed_model}"

    # 1. 知识库级操作 (Knowledge Base Level)
    def create_kb(self):
        """创建知识库：检查是否存在，注册DB并建立物理目录"""
        if kb_exists(self.kb_name):
            logger.warning(f"知识库 '{self.kb_name}' 已存在，跳过创建。")
            return True
        os.makedirs(self.kb_path, exist_ok=True)
        os.makedirs(self.file_path, exist_ok=True)
        os.makedirs(self.json_path, exist_ok=True)
        return add_kb_to_db(self.kb_name, self.kb_info)
    
    # 2. 实验级操作 (Experiment Level)
    def create_experiment(self, chunk_size: int, index_type: str, chunk_overlap: int):
        """创建具体的实验配置及向量库"""
        if experiment_exists(self.exp_name):
            return True

        status = add_experiment_to_db(
            exp_name=self.exp_name,
            kb_name=self.kb_name,
            vs_type=self.vs_type(),
            embed_model=self.embed_model,
            chunk_size=chunk_size,
            index_type=index_type,
            chunk_overlap=chunk_overlap
        )
        return status
    
    def drop_experiment(self):
        """彻底删除当前实验：清理向量库 + 清理DB"""
        if not self.exp_name:
            raise ValueError("未指定 exp_name！")
            
        if not experiment_exists(self.exp_name):
            return True    
        self.do_clear_vs()
        return delete_experiment_from_db(self.exp_name)
    
    # 3. 原文件操作 (File Level)
    def register_file(self, kb_file: KnowledgeFile, is_parsed: bool = False):
        """仅仅将物理文件注册进数据库，不切分、不向量化"""
        if file_exists_in_kb(self.kb_name, kb_file.filename):
            return True
        return add_file_to_db(kb_file, is_parsed=is_parsed)

    def remove_file(self, kb_file: KnowledgeFile, delete_content: bool = False):
        """彻底删除文件：先删各实验的向量库 Chunk，再删数据库，最后删物理文件"""
        
        if not file_exists_in_kb(self.kb_name, kb_file.filename):
            return True
        experiments = list_experiments_from_db(self.kb_name)
        for exp_name_str in experiments:
            chunk_ids = list_chunkId_from_db_by_expName_fileName(exp_name_str, kb_file.filename)
            
            if chunk_ids:
                engine = self.__class__(
                    knowledge_base_name=self.kb_name, 
                    exp_name=exp_name_str, 
                    embed_model=self.embed_model
                )
                engine.del_doc_by_ids(chunk_ids)

        status = delete_file_from_db(kb_file)
        if status and delete_content:
            try:
                if os.path.exists(kb_file.filepath):
                    os.remove(kb_file.filepath)
                
                if os.path.exists(kb_file.jsonPath):
                    os.remove(kb_file.jsonPath)
            except Exception as e:
                pass
        return status
    
    # 4. 文本块 / 向量库操作 (Chunk & Vector Store Level) 核心联动！
    def add_file_to_vs(self, kb_file: KnowledgeFile):
        """
        【核心入口】一站式添加文件：自动注册 -> 获取切片 -> 写入向量引擎 -> 写入DB
        """
        if not experiment_exists(self.exp_name):
            raise ValueError(f"实验'{self.exp_name}'不存在！请先调用create_experiment。")
        
        if not file_exists_in_kb(self.kb_name, kb_file.filename):
            self.register_file(kb_file, is_parsed=True)

        chunk_dicts = getattr(kb_file, 'splited_docs', [])
        
        if not chunk_dicts:
            logger.warning(f"文件 '{kb_file.filename}' 中没有提取到切片数据，跳过入库。")
            return False

        # doc_infos 返回必须带有 'id': "vs-uuid..." 
        doc_infos = self.do_add_doc(chunk_dicts) 
        
        status = add_chunk_to_db_by_expName(
            kb_name=self.kb_name,
            exp_name=self.exp_name,
            file_name=kb_file.filename,
            doc_infos=doc_infos,
        )
        return status
    
    def delete_file_from_vs(self, kb_file: KnowledgeFile):
        """仅从当前的实验向量库中清除该文件的所有切片"""           
        if not experiment_exists(self.exp_name):
            return True
        
        if not file_exists_in_kb(self.kb_name, kb_file.filename):
            return True

        chunk_ids = list_chunkId_from_db_by_expName_fileName(self.exp_name, kb_file.filename)
        if not chunk_ids:
            return True
            
        if self.del_doc_by_ids(chunk_ids):
            delete_chunk_from_db_by_expName_fileName(self.exp_name, kb_file.filename)
            return True
            
        return False
    
    def update_file_in_vs(self, kb_file: KnowledgeFile):
        """自动先删后增"""
        self.delete_file_from_vs(kb_file)
        return self.add_file_to_vs(kb_file)
    
    def search_docs(self, query: str, top_k: int = 10, score_threshold: float = 0.0, **kwargs) -> List:
        """检索必须基于某个实验库"""
        if not self.exp_name:
            raise ValueError("未指定实验名称，无法进行检索！")
            
        if not experiment_exists(self.exp_name):
            raise ValueError(f"实验 '{self.exp_name}' 不存在，无法进行检索操作！")
            
        return self.do_search(query=query, top_k=top_k, score_threshold=score_threshold, **kwargs)

    #子类需实现的抽象方法 (Abstract Methods)
    @abstractmethod
    def vs_type(self) -> str:
        """返回向量库类型，例如 'faiss', 'milvus' 等"""
        pass

    @abstractmethod
    def do_init(self):
        """初始化或加载向量库引擎（绑定 self.exp_name 对应的索引/集合）"""
        pass

    @abstractmethod
    def do_add_doc(self, chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [核心约定]：子类接收基类传入的字典列表，进行向量化并存入引擎。
        输入: [{'metadata': {...}, 'content': '...'}, ...]
        返回: 必须注入引擎生成的 'id'，返回 [{'id': 'xxx', 'metadata': {...}, 'content': '...'}, ...]
        """
        pass

    @abstractmethod
    def del_doc_by_ids(self, ids: List[str]) -> bool:
        """从向量引擎中彻底删除给定 ID 列表对应的向量和Payload"""
        pass

    @abstractmethod
    def do_clear_vs(self):
        """清空/删除当前实验对应的整个向量库实例或物理文件"""
        pass

    @abstractmethod
    def do_search(self, query: str, top_k: int, score_threshold: float) -> List[Dict]:
        """
        返回检索结果，格式建议统一为字典列表:
        [{'id': 'xxx', 'score': 0.89, 'content': '...', 'metadata': {...}}]
        """
        pass

class FaissKBService(KBService):
    def __init__(
        self,
        knowledge_base_name: str = None,
        exp_name: str = None,
        kb_info: str = None,
        index_type: str = Settings.kb_settings.INDEX_TYPE,
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
    ):
        self.index_type = index_type
        self.faiss_device = os.getenv("FINANCE_RAG_FAISS_DEVICE", "cpu").strip().lower()
        self.faiss_gpu_id = int(os.getenv("FINANCE_RAG_FAISS_GPU_ID", "0"))
        self.vector_store = None
        self._bm25_model = None
        self._bm25_docs = None
        super().__init__(
            knowledge_base_name=knowledge_base_name,
            exp_name=exp_name,
            kb_info=kb_info,
            index_type=index_type,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def vs_type(self) -> str:
        return "faiss"

    def do_init(self):
        self.load_vector_store()

    def _get_bm25_model(self):
        """
        内部辅助方法：获取 BM25 模型（带缓存机制）。
        """
        if self._bm25_model is not None and self._bm25_docs is not None:
            return self._bm25_model, self._bm25_docs

        vs = self.load_vector_store()
        docs = list(vs.docstore.values())
        if not docs:
            return None, []
            
        tokenized_corpus = [tokenize_for_search(doc.get("content", "")) for doc in docs]
        self._bm25_model = BM25Okapi(tokenized_corpus)
        self._bm25_docs = docs
        
        return self._bm25_model, self._bm25_docs

    def _get_embedding_client(self):
        """辅助方法：获取 Embedding 实例"""
        provider = os.getenv("FINANCE_RAG_EMBEDDING_PROVIDER", "").strip().lower()
        model_name = os.getenv("FINANCE_RAG_EMBEDDING_MODEL", self.embed_model)

        if self.embed_model == "mock" or provider == "mock":
            dim = int(os.getenv("FINANCE_RAG_MOCK_EMBEDDING_DIM", "384"))
            return MockEmbeddings(dim=dim)

        dimensions_env = os.getenv("FINANCE_RAG_EMBEDDING_DIMENSIONS")
        dimensions = int(dimensions_env) if dimensions_env else getattr(
            Settings.model_settings,
            "DEFAULT_EMBEDDING_DIMENSIONS",
            None,
        )
        batch_size = int(os.getenv("FINANCE_RAG_EMBEDDING_BATCH_SIZE", "10"))

        if provider in {"dashscope", "bailian", "aliyun"} or model_name.startswith("text-embedding-v"):
            return AliyunBailianEmbeddings(
                model_name=model_name,
                api_key=os.getenv("FINANCE_RAG_EMBEDDING_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
                base_url=os.getenv(
                    "FINANCE_RAG_EMBEDDING_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                dimensions=dimensions or 1024,
                batch_size=batch_size,
                timeout=int(os.getenv("FINANCE_RAG_EMBEDDING_TIMEOUT", "120")),
                max_retries=int(os.getenv("FINANCE_RAG_EMBEDDING_MAX_RETRIES", "3")),
            )

        base_url = os.getenv(
            "FINANCE_RAG_EMBEDDING_BASE_URL",
            Settings.model_settings.DEFAULT_EMBEDDING_MODEL_BASE_URL,
        )
        api_key = os.getenv("FINANCE_RAG_EMBEDDING_API_KEY", os.getenv("DASHSCOPE_API_KEY", "EMPTY"))
        return OpenAICompatibleEmbeddings(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            dimensions=dimensions,
            batch_size=batch_size,
            timeout=int(os.getenv("FINANCE_RAG_EMBEDDING_TIMEOUT", "120")),
            max_retries=int(os.getenv("FINANCE_RAG_EMBEDDING_MAX_RETRIES", "3")),
        )

    def load_vector_store(self):
        """加载 FAISS 引擎：不存在则新建""" 
        if self.vector_store is not None:
            return self.vector_store
            
        embed_client = self._get_embedding_client()
        index_path = os.path.join(self.vs_path, "index.faiss")
        pkl_path = os.path.join(self.vs_path, "index.pkl")

        if os.path.exists(index_path) and os.path.exists(pkl_path):
            self.vector_store = NativeFAISS.load_local(
                folder_path=self.vs_path, 
                embedding_model=embed_client, 
                index_name="index",
                device=self.faiss_device,
                gpu_id=self.faiss_gpu_id,
            )
        else:
            if os.path.exists(index_path) or os.path.exists(pkl_path):
                logger.warning(f"检测到实验 {self.exp_name} 的向量库文件不完整，将重新初始化。")

            dim = getattr(embed_client, "dimension", None)
            if not dim:
                test_embed = embed_client.embed_query("test_dim")
                dim = len(test_embed)
            
            index = NativeFAISS.create_index(dim=dim, index_type=self.index_type)
            self.vector_store = NativeFAISS(
                embedding_model=embed_client,
                index=index,
                index_type=self.index_type,
                device=self.faiss_device,
                gpu_id=self.faiss_gpu_id,
            )
            
        return self.vector_store

    def do_add_doc(self, chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """向 FAISS 灌入切片字典"""
        vs = self.load_vector_store()
        ids = vs.add_documents(chunk_dicts) 
            
        for i, doc in enumerate(chunk_dicts):
            doc["id"] = ids[i]

        self._bm25_model = None
        self._bm25_docs = None
            
        return chunk_dicts

    def do_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        search_mode: str = "rrf",
        vector_search_method: str = "similarity", # 向量检索算法: similarity, mmr
        mmr_lambda: float = 0.5,
        is_eval: bool = False,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_c: int = 60,
    ) -> List[Dict[str, Any]]:
        """执行多路检索与融合"""
        vs = self.load_vector_store()
        
        # 如果是融合模式，底层召回数建议稍微放大，为融合留出空间
        actual_k = top_k * 2 if search_mode in ["weight", "rrf"] else top_k

        if vector_search_method == "mmr":
            raw_vector_results = vs.max_marginal_relevance_search(
                query=query,
                k=actual_k,
                fetch_k=top_k * 4,
                lambda_mult=mmr_lambda
            )
        else:
            raw_vector_results = vs.similarity_search_with_score(query, k=actual_k)
        vector_docs = []
        for doc_dict, l2_distance in raw_vector_results:
            sim_score = 1 / (1 + l2_distance) 
            doc_copy = doc_dict.copy()
            doc_copy["vector_score"] = sim_score
            vector_docs.append(doc_copy)

        bm25_docs = []
        if search_mode in ["bm25", "weight", "rrf"]:
            bm25_model, all_docs = self._get_bm25_model()
            if bm25_model:
                tokenized_query = tokenize_for_search(query)
                bm25_scores = bm25_model.get_scores(tokenized_query)
                
                top_n_idx = top_score_indices(bm25_scores, top_k * 2)
                for idx in top_n_idx:
                    if bm25_scores[idx] > 0:
                        doc_copy = all_docs[idx].copy()
                        doc_copy["bm25_score"] = bm25_scores[idx]
                        bm25_docs.append(doc_copy)

        final_results = {}

        if search_mode == "vector":
            for doc in vector_docs:
                doc["score"] = doc["vector_score"]
                final_results[doc["id"]] = doc

        elif search_mode == "bm25":
            for doc in bm25_docs:
                doc["score"] = doc["bm25_score"]
                final_results[doc["id"]] = doc

        elif search_mode == "weight":
            def min_max_scale(docs, score_key):
                scores = [d[score_key] for d in docs]
                if not scores: return {}
                min_s, max_s = min(scores), max(scores)
                scaled_dict = {}
                for d in docs:
                    norm_score = (d[score_key] - min_s) / (max_s - min_s) if max_s > min_s else 1.0
                    scaled_dict[d["id"]] = norm_score
                return scaled_dict

            v_norm_scores = min_max_scale(vector_docs, "vector_score")
            b_norm_scores = min_max_scale(bm25_docs, "bm25_score")

            all_ids = set(v_norm_scores.keys()).union(set(b_norm_scores.keys()))
            doc_map = {d["id"]: d for d in vector_docs + bm25_docs}

            for doc_id in all_ids:
                v_score = v_norm_scores.get(doc_id, 0.0)
                b_score = b_norm_scores.get(doc_id, 0.0)
                final_score = (v_score * vector_weight) + (b_score * bm25_weight)
                
                merged_doc = doc_map[doc_id].copy()
                merged_doc["score"] = final_score
                final_results[doc_id] = merged_doc

        elif search_mode == "rrf":
            rrf_scores = {}
            doc_map = {d["id"]: d for d in vector_docs + bm25_docs}

            for rank, doc in enumerate(vector_docs, start=1):
                rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (vector_weight / (rank + rrf_c))

            for rank, doc in enumerate(bm25_docs, start=1):
                rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (bm25_weight / (rank + rrf_c))

            for doc_id, rrf_score in rrf_scores.items():
                merged_doc = doc_map[doc_id].copy()
                merged_doc["score"] = rrf_score
                final_results[doc_id] = merged_doc

        filtered_results = []
        for doc_id, doc in final_results.items():
            if is_eval or doc["score"] >= score_threshold:
                filtered_results.append(doc)

        filtered_results.sort(key=lambda x: x["score"], reverse=True)
        return filtered_results[:top_k]

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        """根据 chunk ID 精确删除"""
        if not ids:
            return True
            
        vs = self.load_vector_store()
        vs.delete_by_ids(ids)
        self._bm25_model = None
        self._bm25_docs = None
        return True

    def do_clear_vs(self):
        """清空当前实验的 FAISS 库"""
        self.vector_store = None

        self._bm25_model = None
        self._bm25_docs = None
        
        try:
            shutil.rmtree(self.vs_path)
        except Exception:
            pass
        os.makedirs(self.vs_path, exist_ok=True)

class KBServiceFactory:
    @staticmethod
    def get_service_by_name(
        knowledge_base_name: str,
        exp_name: str,
        kb_info: str,
        index_type: str,
        embed_model: str,
        chunk_size: int,
        chunk_overlap: int,
        vector_store_type: str,
    ):
        """
        统一获取知识库服务
        """
        common_kwargs = {
            "knowledge_base_name": knowledge_base_name,
            "exp_name": exp_name,
            "kb_info": kb_info,
            "index_type": index_type,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        # FAISS
        if vector_store_type == SupportedVSType.FAISS:
            return FaissKBService(**common_kwargs)

        raise ValueError(f"不支持的向量库类型: {vector_store_type}")
