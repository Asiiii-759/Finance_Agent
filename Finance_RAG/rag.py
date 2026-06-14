from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Finance_RAG.parser_chunk_search.chunker import (
    KnowledgeFile,
    files2docs_in_thread,
    get_file_path,
)
from Finance_RAG.parser_chunk_search.kb_service import (
    KBService,
    KBServiceFactory,
    SupportedVSType,
)
from Finance_RAG.providers.rerank import NoopRerankProvider, RerankProvider
from Finance_RAG.settings import Settings
from Finance_RAG.utils import build_logger


_KB_ENGINE_CACHE: Dict[Tuple[str, str, str], KBService] = {}

logger = build_logger()


def get_kb_engine(
    knowledge_base_name: str = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE,
    exp_name: str = Settings.kb_settings.DEFAULT_EXMPERIMENT,
    kb_info: Optional[str] = None,
    index_type: str = Settings.kb_settings.INDEX_TYPE,
    embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
    vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
) -> KBService:
    """
    获取知识库引擎。

    第一版仍保留内存缓存，但缓存 key 包含向量库类型，避免 FAISS/Milvus 后端混用。
    """
    kb_info = kb_info or Settings.kb_settings.KB_INFO.get(
        knowledge_base_name,
        f"关于 {knowledge_base_name} 的知识库",
    )
    cache_key = (knowledge_base_name, exp_name, vector_store_type)

    if cache_key not in _KB_ENGINE_CACHE:
        logger.info(
            f"[Cache Miss] 加载知识库: KB={knowledge_base_name}, "
            f"Exp={exp_name}, VS={vector_store_type}, Embed={embed_model}"
        )
        _KB_ENGINE_CACHE[cache_key] = KBServiceFactory.get_service_by_name(
            knowledge_base_name=knowledge_base_name,
            exp_name=exp_name,
            kb_info=kb_info,
            index_type=index_type,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_store_type=vector_store_type,
        )
    else:
        logger.info(f"[Cache Hit] KB={knowledge_base_name}, Exp={exp_name}, VS={vector_store_type}")

    return _KB_ENGINE_CACHE[cache_key]


def release_kb_engine(
    knowledge_base_name: str,
    exp_name: str,
    vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
) -> None:
    cache_key = (knowledge_base_name, exp_name, vector_store_type)
    if cache_key in _KB_ENGINE_CACHE:
        del _KB_ENGINE_CACHE[cache_key]
        logger.info(f"已释放知识库引擎: KB={knowledge_base_name}, Exp={exp_name}, VS={vector_store_type}")


def _parse_pdf_to_json(kb_file: KnowledgeFile) -> bool:
    """
    JSON 缺失时才触发 PDF 解析。

    这里懒加载解析 provider，避免本地没有 OCR/vLLM 或 API token 时连导入 RAG 模块都失败。
    """
    parser_provider = os.getenv("FINANCE_RAG_PARSER_PROVIDER", "local_paddleocr").strip().lower()
    try:
        if parser_provider in {"paddleocr_api", "api", "paddle_api"}:
            from Finance_RAG.parsers.paddle_ocr_api import PaddleOcrApiParser

            parser = PaddleOcrApiParser()
        else:
            from Finance_RAG.parser_chunk_search.pdf_parser import StructuredDocumentBuilder

            parser = StructuredDocumentBuilder()
    except Exception as exc:
        raise RuntimeError(
            "解析 JSON 不存在，且当前环境无法加载 PDF 解析器。"
            "请先提供 raw_resolve 中的同名 JSON，或安装/配置 OCR 解析依赖/API token。"
        ) from exc

    parsed_data = parser.parse_pdf(kb_file.filepath, save_json=True)
    return bool(parsed_data)


def _prepare_kb_file(file_name: str, knowledge_base_name: str) -> Tuple[Optional[KnowledgeFile], Optional[str]]:
    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)

    if os.path.exists(kb_file.jsonPath):
        return kb_file, None

    if not os.path.exists(kb_file.filepath):
        return None, "content 目录中未找到 PDF，raw_resolve 中也没有同名 JSON"

    logger.info(f"[{file_name}] raw_resolve JSON 不存在，尝试解析 PDF")
    try:
        if not _parse_pdf_to_json(kb_file):
            return None, "PDF 解析未产出有效 JSON"
    except Exception as exc:
        return None, f"PDF 解析失败: {exc}"

    if not os.path.exists(kb_file.jsonPath):
        return None, "PDF 解析完成但未找到预期 JSON"

    return kb_file, None


def update_docs(
    file_names: List[str],
    kb_engine: KBService | None = None,
    knowledge_base_name: str = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE,
    exp_name: str = Settings.kb_settings.DEFAULT_EXMPERIMENT,
    embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
    chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
    not_refresh_vs_cache: bool = False,
) -> Dict[str, Any]:
    """
    将已存在的 PDF/JSON 更新进向量库。

    第一版优先复用 raw_resolve 中的 JSON；只有 JSON 缺失时才触发 PDF 解析。
    """
    logger.info(f"开始更新知识库 {knowledge_base_name} / {exp_name}: {file_names}")

    failed_files: Dict[str, str] = {}
    success_files: List[str] = []
    kb_files_to_chunk: List[KnowledgeFile] = []

    try:
        kb_engine = kb_engine or get_kb_engine(
            knowledge_base_name=knowledge_base_name,
            exp_name=exp_name,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_store_type=vector_store_type,
        )
    except Exception as exc:
        logger.exception(f"知识库服务获取失败: {exc}")
        return {"msg": f"获取引擎失败: {exc}", "success_files": [], "failed_files": failed_files}

    for file_name in file_names:
        kb_file, error = _prepare_kb_file(file_name, knowledge_base_name)
        if error:
            failed_files[file_name] = error
            continue
        kb_files_to_chunk.append(kb_file)

    if kb_files_to_chunk:
        logger.info(f"开始切块并入库，共 {len(kb_files_to_chunk)} 个文件")
        for status, result in files2docs_in_thread(
            files=kb_files_to_chunk,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            if not status:
                _, file_name, error_msg = result
                failed_files[file_name] = error_msg
                continue

            kb_name, file_name, splited_docs = result
            if not splited_docs:
                failed_files[file_name] = "文档切块结果为空"
                continue

            current_kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
            current_kb_file.splited_docs = splited_docs

            try:
                if kb_engine.update_file_in_vs(current_kb_file):
                    success_files.append(file_name)
                else:
                    failed_files[file_name] = "向量化入库失败"
            except Exception as exc:
                logger.exception(f"[{file_name}] 入库异常: {exc}")
                failed_files[file_name] = f"入库异常: {exc}"

    if not not_refresh_vs_cache and success_files and hasattr(kb_engine, "vector_store"):
        vector_store = getattr(kb_engine, "vector_store", None)
        if hasattr(vector_store, "save_local"):
            logger.info(f"保存 FAISS 索引: {kb_engine.vs_path}")
            vector_store.save_local(folder_path=kb_engine.vs_path, index_name="index")

    msg = f"更新完成。成功 {len(success_files)} 个，失败 {len(failed_files)} 个。"
    logger.info(msg)
    return {"msg": msg, "success_files": success_files, "failed_files": failed_files}


def upload_docs(
    source_file_paths: List[str],
    kb_engine: KBService | None = None,
    knowledge_base_name: str = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE,
    exp_name: str = Settings.kb_settings.DEFAULT_EXMPERIMENT,
    embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
    override: bool = False,
    to_vector_store: bool = True,
    not_refresh_vs_cache: bool = False,
    chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
) -> Dict[str, Any]:
    """
    拷贝外部 PDF 到知识库 content 目录，并可选触发入库。
    """
    failed_files: Dict[str, str] = {}
    uploaded_filenames: List[str] = []

    for source_path in source_file_paths:
        path_obj = Path(source_path)
        if not path_obj.exists() or not path_obj.is_file():
            failed_files[source_path] = "源文件不存在或不是文件"
            continue

        filename = path_obj.name
        target_path = get_file_path(knowledge_base_name, filename)

        try:
            if str(path_obj.resolve()) != str(Path(target_path).resolve()):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                if os.path.exists(target_path) and not override:
                    if os.path.getsize(target_path) == os.path.getsize(source_path):
                        logger.warning(f"文件 {filename} 已存在且大小相同，跳过拷贝")
                        uploaded_filenames.append(filename)
                        continue
                    failed_files[filename] = "目标目录已有同名文件，且未开启 override"
                    continue

                shutil.copy2(source_path, target_path)
                logger.info(f"已拷贝文件到知识库: {filename}")

            if filename not in uploaded_filenames:
                uploaded_filenames.append(filename)
        except Exception as exc:
            failed_files[filename] = f"拷贝文件失败: {exc}"

    if to_vector_store and uploaded_filenames:
        update_result = update_docs(
            file_names=uploaded_filenames,
            kb_engine=kb_engine,
            knowledge_base_name=knowledge_base_name,
            exp_name=exp_name,
            embed_model=embed_model,
            vector_store_type=vector_store_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            not_refresh_vs_cache=not_refresh_vs_cache,
        )
        update_result["failed_files"].update(failed_files)
        return update_result

    return {
        "msg": "文件上传完成，未执行向量化",
        "uploaded_filenames": uploaded_filenames,
        "failed_files": failed_files,
    }


def delete_docs(
    file_names: List[str],
    kb_engine: KBService | None = None,
    knowledge_base_name: str = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE,
    exp_name: str = Settings.kb_settings.DEFAULT_EXMPERIMENT,
    embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
    delete_content: bool = False,
    not_refresh_vs_cache: bool = False,
) -> Dict[str, Any]:
    failed_files: Dict[str, str] = {}
    success_files: List[str] = []

    try:
        kb_engine = kb_engine or get_kb_engine(
            knowledge_base_name=knowledge_base_name,
            exp_name=exp_name,
            embed_model=embed_model,
            vector_store_type=vector_store_type,
        )
    except Exception as exc:
        return {"msg": f"获取引擎失败: {exc}", "success_files": [], "failed_files": failed_files}

    for file_name in file_names:
        try:
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            status = (
                kb_engine.remove_file(kb_file, delete_content=True)
                if delete_content
                else kb_engine.delete_file_from_vs(kb_file)
            )
            if status:
                success_files.append(file_name)
            else:
                failed_files[file_name] = "删除操作失败"
        except Exception as exc:
            logger.exception(f"[{file_name}] 删除异常: {exc}")
            failed_files[file_name] = f"删除异常: {exc}"

    if not not_refresh_vs_cache and success_files and hasattr(kb_engine, "vector_store"):
        vector_store = getattr(kb_engine, "vector_store", None)
        if hasattr(vector_store, "save_local"):
            vector_store.save_local(folder_path=kb_engine.vs_path, index_name="index")

    msg = f"删除完成。成功 {len(success_files)} 个，失败 {len(failed_files)} 个。"
    return {"msg": msg, "success_files": success_files, "failed_files": failed_files}


def retrieve_documents(
    query: str,
    kb_engine: KBService | None = None,
    knowledge_base_name: str = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE,
    exp_name: str = Settings.kb_settings.DEFAULT_EXMPERIMENT,
    embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
    top_k: int = Settings.kb_settings.VECTOR_SEARCH_TOP_K,
    search_mode: str = "rrf",
    score_threshold: float = 0.0,
    **search_kwargs: Any,
) -> List[Dict[str, Any]]:
    kb_engine = kb_engine or get_kb_engine(
        knowledge_base_name=knowledge_base_name,
        exp_name=exp_name,
        embed_model=embed_model,
        vector_store_type=vector_store_type,
    )
    logger.info(f"检索 query: {query}")
    return kb_engine.search_docs(
        query=query,
        top_k=top_k,
        search_mode=search_mode,
        score_threshold=score_threshold,
        **search_kwargs,
    )


def noop_rerank_documents(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    logger.info(f"Noop rerank，保留前 {top_n} 个结果")
    return NoopRerankProvider().rerank(query=query, docs=retrieved_docs, top_n=top_n)


def rerank_documents(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    top_n: int = 3,
    provider: RerankProvider | None = None,
) -> List[Dict[str, Any]]:
    provider = provider or NoopRerankProvider()
    return provider.rerank(query=query, docs=retrieved_docs, top_n=top_n)


def format_docs_for_llm(docs: List[Dict[str, Any]]) -> str:
    context_parts = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {})
        source = meta.get("file_name", "未知来源")
        title = meta.get("paragraph_title") or meta.get("doc_title") or "无标题"
        page = meta.get("source_page")
        content = doc.get("content", "").strip()

        page_text = f"\n- 页码: {page}" if page is not None else ""
        context_parts.append(
            f"### [参考文档 {index}]\n"
            f"- 来源: {source}{page_text}\n"
            f"- 标题: {title}\n"
            f"- 内容: {content}\n"
        )

    return "\n".join(context_parts)


def build_llm_prompt(query: str, context_str: str) -> str:
    return f"""你是一个专业的金融分析助手。请只基于以下参考文档回答用户问题。
如果参考文档中没有相关信息，请明确说明“根据提供的资料无法回答”，不要编造数据。

【参考文档】
{context_str}

【用户问题】
{query}

【回答】
"""


if __name__ == "__main__":
    engine = get_kb_engine(embed_model="mock")
    docs = retrieve_documents(
        kb_engine=engine,
        query="汽车电子业务增长情况如何？",
        top_k=5,
        search_mode="rrf",
    )
    print(format_docs_for_llm(noop_rerank_documents("", docs, top_n=3)))
