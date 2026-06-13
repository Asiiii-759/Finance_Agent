import os
import uuid
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional

class NativeFAISS:
    def __init__(
        self, 
        embedding_model: Any, 
        index: faiss.Index, 
        index_type: str,
        device: str = "cpu",
        gpu_id: int = 0,
        docstore: Optional[Dict[str, Dict[str, Any]]] = None, 
        index_to_docstore_id: Optional[Dict[int, str]] = None
    ):
        """
        初始化 NativeFAISS。
        :param embedding_model: 你的 VllmBgeEmbeddings 实例
        :param index: faiss 的 Index 对象
        :param docstore: 存储 {uuid: 文档字典}
        :param index_to_docstore_id: 存储 {faiss_id: uuid}
        """
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.device = device
        self.gpu_id = gpu_id
        self.index = self._move_index_to_device(index)
        self.docstore = docstore if docstore is not None else {}
        self.index_to_docstore_id = index_to_docstore_id if index_to_docstore_id is not None else {}

    @staticmethod
    def gpu_available() -> bool:
        return hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")

    def _move_index_to_device(self, index: faiss.Index) -> faiss.Index:
        if self.device == "cpu":
            return index
        if self.device != "gpu":
            raise ValueError(f"不支持的 FAISS device: {self.device}")
        if not self.gpu_available():
            raise RuntimeError(
                "当前 Python 环境中的 faiss 不包含 GPU API。"
                "请在 Linux/WSL2 conda 环境安装官方 faiss-gpu 后再设置 FINANCE_RAG_FAISS_DEVICE=gpu。"
            )
        resources = faiss.StandardGpuResources()
        try:
            return faiss.index_cpu_to_gpu(resources, self.gpu_id, index)
        except Exception as exc:
            raise RuntimeError(
                f"无法将 FAISS {self.index_type} 索引迁移到 GPU。"
                "请确认该索引类型被 faiss-gpu 支持，或改用 Flat/IVF。"
            ) from exc

    def _cpu_index_for_persistence(self) -> faiss.Index:
        if self.device == "gpu":
            if not hasattr(faiss, "index_gpu_to_cpu"):
                raise RuntimeError("当前 faiss 缺少 index_gpu_to_cpu，无法保存 GPU 索引")
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    @classmethod
    def create_index(cls, dim: int, index_type: str = "Flat", **kwargs) -> faiss.Index:
        """
        创建一个 FAISS 索引。
        :param dim: 向量维度
        :param index_type: 索引类型 "Flat", "HNSW", "IVF"
        """
        if index_type == "Flat":
            return faiss.IndexFlatL2(dim)
        
        elif index_type == "HNSW":
            m = kwargs.get("m", 32)
            return faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
            
        elif index_type == "IVF":
            nlist = kwargs.get("nlist", 200)
            quantizer = faiss.IndexFlatL2(dim)
            return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")

    def add_documents(self, dict_list: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """
        向向量库中添加你的自定义字典数据
        """
        if not dict_list:
            return []

        embeddings = self.embedding_model.embed_dict_list(dict_list, content_key="content")
        vector_array = np.array(embeddings, dtype=np.float32)

        if not self.index.is_trained:
            print("检测到索引尚未训练，正在进行训练...")
            self.index.train(vector_array)

        ids = ids or [str(uuid.uuid4()) for _ in dict_list]
        if len(ids) != len(dict_list):
            raise ValueError("ids 数量必须与文档数量一致")
        starting_len = len(self.index_to_docstore_id)

        self.index.add(vector_array)

        for j, (doc_id, doc_dict) in enumerate(zip(ids, dict_list)):
            self.docstore[doc_id] = doc_dict
            self.index_to_docstore_id[starting_len + j] = doc_id

        return ids
    
    def delete_by_ids(self, ids_to_delete: List[str]) -> bool:
        """
        从 FAISS 和 Docstore 中删除指定 ID 的记录。
        通过过滤保留项并重建索引来实现。
        """
        if not ids_to_delete:
            return True
            
        ids_set = set(ids_to_delete)
        
        remaining_items = [
            (doc_id, doc) for doc_id, doc in self.docstore.items()
            if doc_id not in ids_set
        ]
        
        if len(remaining_items) == len(self.docstore):
            return True
            
        cpu_index = self._cpu_index_for_persistence()
        dim = cpu_index.d
        self.index = self.create_index(dim, index_type=self.index_type)
        self.index = self._move_index_to_device(self.index)
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        
        if remaining_items:
            remaining_ids = [item[0] for item in remaining_items]
            remaining_docs = [item[1] for item in remaining_items]
            self.add_documents(remaining_docs, ids=remaining_ids)
            
        return True

    def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        相似度搜索并返回分数 (L2距离，越小越相似)
        """
        query_embedding = self.embedding_model.embed_query(query)
        vector = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(vector, k)

        results = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                continue
            
            doc_id = self.index_to_docstore_id[i]
            doc_dict = self.docstore[doc_id]
            score = scores[0][j]
            
            results.append((doc_dict, score))
            
        return results

    def similarity_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        仅返回相似的文档列表，不返回分数
        """
        results = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in results]
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        最大边际相关性检索（MMR）
        兼顾：和查询相似 + 文档之间不重复
        
        :param query: 查询文本
        :param k: 最终返回多少条结果
        :param fetch_k: 先从向量库取多少条候选文档参与 MMR 计算
        :param lambda_mult: 1=只看相似度，0=只看多样性，0.5=平衡
        :return: 返回一个列表，列表元素为 (文档字典, 相似度分数) 的元组
        """
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_vector, fetch_k)

        candidates = []
        valid_indices = []

        for j, idx in enumerate(indices[0]):
            if idx == -1:
                continue
                
            doc_id = self.index_to_docstore_id[idx]
            doc = self.docstore[doc_id]
            score = scores[0][j]
            candidates.append((doc, score))
            valid_indices.append(idx)

        if not candidates:
            return []

        candidate_vectors = self.index.reconstruct_batch(np.array(valid_indices))
        candidate_vectors = np.array(candidate_vectors, dtype=np.float32)
        selected_idxs = self._mmr_ranking(
            query_vector=query_vector,
            candidate_vectors=candidate_vectors,
            k=k,
            lambda_mult=lambda_mult
        )
        
        return [candidates[i] for i in selected_idxs]

    def _mmr_ranking(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        k: int,
        lambda_mult: float
    ) -> List[int]:
        """MMR 内部排序逻辑"""
        n = len(candidate_vectors)
        if n == 0:
            return []

        similarity_to_query = np.dot(candidate_vectors, query_vector.T).flatten()
        similarity_matrix = np.dot(candidate_vectors, candidate_vectors.T)

        selected = []
        candidates_idx = list(range(n))

        while len(selected) < k and candidates_idx:
            best_score = -np.inf
            best_idx = -1

            for idx in candidates_idx:
                sim_query = similarity_to_query[idx]
                sim_selected = max([similarity_matrix[idx][s] for s in selected], default=0)
                score = lambda_mult * sim_query - (1 - lambda_mult) * sim_selected

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1:
                break
            selected.append(best_idx)
            candidates_idx.remove(best_idx)

        return selected

    def save_local(self, folder_path: str, index_name: str = "index"):
        """
        将 FAISS 索引和文本数据保存到本地
        """
        os.makedirs(folder_path, exist_ok=True)
        index_path = os.path.join(folder_path, f"{index_name}.faiss")
        faiss.write_index(self._cpu_index_for_persistence(), index_path)

        pkl_path = os.path.join(folder_path, f"{index_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "docstore": self.docstore,
                    "index_to_docstore_id": self.index_to_docstore_id,
                    "index_type": self.index_type,
                    "device": self.device,
                    "gpu_id": self.gpu_id,
                },
                f,
            )
            
        print(f"数据已成功保存至 {folder_path}")

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embedding_model: Any,
        index_name: str = "index",
        device: str = "cpu",
        gpu_id: int = 0,
    ):
        """
        从本地加载 FAISS 索引和文本数据
        """
        index_path = os.path.join(folder_path, f"{index_name}.faiss")
        pkl_path = os.path.join(folder_path, f"{index_name}.pkl")

        if not os.path.exists(index_path) or not os.path.exists(pkl_path):
            raise FileNotFoundError(f"在 {folder_path} 中未找到对应的 .faiss 或 .pkl 文件。")

        index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            docstore = payload["docstore"]
            index_to_docstore_id = payload["index_to_docstore_id"]
            index_type = payload.get("index_type", "Flat")
        else:
            docstore, index_to_docstore_id = payload
            index_type = "Flat"

        return cls(
            embedding_model=embedding_model,
            index=index,
            index_type=index_type,
            device=device,
            gpu_id=gpu_id,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
