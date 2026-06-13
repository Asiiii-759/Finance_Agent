import json
import os
import re
import sentencepiece as spm
from pathlib import Path
from itertools import groupby
from urllib.parse import urlencode
from Finance_RAG.settings import Settings
from Finance_RAG.utils import build_logger
from typing import Callable, Dict, Generator, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = build_logger()


def validate_kb_name(knowledge_base_id: str) -> bool:
    if "../" in knowledge_base_id:
        return False
    return True

def get_kb_path(knowledge_base_name: str):
    return os.path.join(Settings.basic_settings.KB_ROOT_PATH, knowledge_base_name)

def get_raw_resolve_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "raw_resolve")

def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")

def get_vs_path(knowledge_base_name: str, exp_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", exp_name)

def get_json_path(knowledge_base_name: str, file_name: str = None):
    if file_name is None:
        return get_raw_resolve_path(knowledge_base_name)

    raw_path = Path(get_raw_resolve_path(knowledge_base_name)).resolve()
    file_path = (raw_path / file_name).resolve()
    if str(file_path).startswith(str(raw_path)):
        return str(file_path)
    raise ValueError(f"非法 JSON 文件路径: {file_name}")

def get_file_path(knowledge_base_name: str, file_name: str):
    raw_path = Path(get_doc_path(knowledge_base_name)).resolve()
    file_path = (raw_path / file_name).resolve()
    if str(file_path).startswith(str(raw_path)):
        return str(file_path)

def list_kbs_from_folder():
    return [
        f
        for f in os.listdir(Settings.basic_settings.KB_ROOT_PATH)
        if os.path.isdir(os.path.join(Settings.basic_settings.KB_ROOT_PATH, f))
    ]

def list_files_from_folder(kb_name: str):
    raw_path = get_doc_path(kb_name)
    result = []
    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False
    with os.scandir(raw_path) as it:
        for entry in it:
            if is_skiped_path(entry.path):
                continue

            if entry.is_file():
                file_path = Path(
                    os.path.relpath(entry.path, raw_path)
                ).as_posix()
                result.append(file_path)

    return result

def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps

class BaseBlockSplitter:
    
    _RE_HTML_TABLE = re.compile(r'(.*?)(<table[^>]*>)(.*?)(</table>)(.*)', re.IGNORECASE | re.DOTALL)
    _RE_HTML_TR = re.compile(r'<tr[^>]*>.*?</tr>', re.IGNORECASE | re.DOTALL)
    _RE_MD_TABLE = re.compile(r'(.*?内容：\s*)(.*?)(\s*脚注.*|$)', re.DOTALL)
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        spm_model_path: str = Settings.kb_settings.TOKENIZER_FILE
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if chunk_overlap >= chunk_size:
            adjusted_overlap = max(0, chunk_size // 4)
            logger.warning(
                f"chunk_overlap({chunk_overlap}) >= chunk_size({chunk_size})，"
                f"已自动调整为 {adjusted_overlap}"
            )
            chunk_overlap = adjusted_overlap

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold = chunk_size * 0.8
        if spm_model_path and os.path.exists(spm_model_path):
            self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
            self.length_function = lambda x: len(self.sp.encode(x))
            self.get_tail_text = self._get_tail_by_token
        else:
            if spm_model_path:
                logger.warning(f"Tokenizer 文件不存在，回退到字符长度切分: {spm_model_path}")
            self.length_function = len
            self.get_tail_text = lambda text, size: text[-size:] if size > 0 else ""
            
    def _get_tail_by_token(self, text: str, token_size: int) -> str:
        """辅助函数：按 token 数量精准截取文本尾部作为 overlap"""
        if token_size <= 0 or not text:
            return ""
        tokens = self.sp.encode(text)
        if len(tokens) <= token_size:
            return text
        return self.sp.decode(tokens[-token_size:])

    def _find_window_end_by_length(self, text: str, start: int, max_length: int) -> int:
        """在原始字符边界上找一个最大 end，使 text[start:end] 的长度估算不超过 max_length。"""
        if start >= len(text):
            return start

        low = start + 1
        high = len(text)
        best = low
        while low <= high:
            mid = (low + high) // 2
            if self.length_function(text[start:mid]) <= max_length:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return max(best, start + 1)

    def _find_overlap_start_by_length(self, text: str, start: int, end: int) -> int:
        """在原始字符边界上找尽量长、但长度估算不超过 chunk_overlap 的后缀起点。"""
        if self.chunk_overlap <= 0:
            return end

        low = start
        high = end
        best = end
        while low <= high:
            mid = (low + high) // 2
            if self.length_function(text[mid:end]) <= self.chunk_overlap:
                best = mid
                high = mid - 1
            else:
                low = mid + 1

        return best

    def _split_long_span_text(self, text: str, global_start: int) -> List[Tuple[str, int, int]]:
        """
        按原文字符边界切分，并返回全局字符坐标。

        tokenizer 只用于估算窗口长度，不参与 token decode -> find 原文的反向定位。
        """
        chunks = []
        text_len = len(text)
        current_pos = 0

        while current_pos < text_len:
            end_pos = self._find_window_end_by_length(text, current_pos, self.chunk_size)
            chunk_text = text[current_pos:end_pos]
            chunks.append((
                chunk_text,
                global_start + current_pos,
                global_start + end_pos - 1,
            ))

            if end_pos >= text_len:
                break

            next_pos = self._find_overlap_start_by_length(text, current_pos, end_pos)
            if next_pos <= current_pos:
                next_pos = current_pos + 1
            current_pos = next_pos

        return chunks

    def _ensure_block_coordinates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """确保每个 block 都有基于原始 block_content 串联文本的 1-based 闭区间坐标。"""
        needs_patch = any(
            block.get("global_start") is None or block.get("global_end") is None
            for block in blocks
        )
        if not needs_patch:
            return blocks

        patched = []
        current_global_idx = 1
        for block in blocks:
            copied = block.copy()
            content = copied.get("block_content", "") or ""
            content_len = len(content)
            if content_len > 0:
                copied["global_start"] = current_global_idx
                copied["global_end"] = current_global_idx + content_len - 1
                current_global_idx = copied["global_end"] + 1
            else:
                copied["global_start"] = current_global_idx
                copied["global_end"] = current_global_idx - 1
            patched.append(copied)
        return patched
    
    def _split_table_and_chart(self, text: str, global_start: int, global_end: int) -> Tuple[List[str], List[Tuple[int, int]], str]:
        """
        返回: (切分后的字符串列表, 对应的全局坐标元组列表, note表头信息)
        在满足条件不切分时，直接返回原文本、原坐标和空note
        """
        total_len = self.length_function(text)
        if total_len <= self.chunk_size:
            return [text], [(global_start, global_end)], ""
        
        text_lower = text.lower()
        if "<table>" in text_lower and "</table>" in text_lower:
            return self._split_html_table(text, total_len, global_start, global_end)

        chunks = self._split_long_span_text(text, global_start)
        note = text.splitlines()[0] if text.splitlines() else ""
        return [item[0] for item in chunks], [(item[1], item[2]) for item in chunks], note

    def _split_html_table(self, text: str, total_len: int, global_start: int, global_end: int) -> Tuple[List[str], List[Tuple[int, int]], str]:
        match = self._RE_HTML_TABLE.search(text)
        if not match:
            return [text], [(global_start, global_end)], ""
            
        prefix, table_start, table_inner, table_end, suffix = match.groups()
        rows = self._RE_HTML_TR.findall(table_inner)
        
        if len(rows) <= 1:
            return [text], [(global_start, global_end)], ""
            
        header_row = rows[0]
        data_rows = rows[1:]
        if any(self.length_function(row) > self.chunk_size for row in rows):
            logger.warning("检测到 HTML 表格单行超过 chunk_size，整段表格按原文字符坐标安全拆分")
            chunks = self._split_long_span_text(text, global_start)
            return [item[0] for item in chunks], [(item[1], item[2]) for item in chunks], header_row

        head_wrap = prefix + table_start
        tail_wrap = table_end + suffix
        
        return self._distribute_rows(head_wrap, header_row, tail_wrap, data_rows, total_len, global_start, global_end)

    def _split_markdown_table(self, text: str, total_len: int, global_start: int, global_end: int) -> Tuple[List[str], List[Tuple[int, int]], str]:
        """处理保留了换行符 (\n) 的纯文本/Markdown表格（按行优雅切分并共享表头）"""
        match = self._RE_MD_TABLE.search(text)
        if not match:
            prefix, table_inner, suffix = "", text, ""
        else:
            prefix, table_inner, suffix = match.groups()
            
        lines = table_inner.strip().split('\n')
        if len(lines) <= 1:
            return [text], [(global_start, global_end)], ""
            
        header_row = lines[0] + "\n"
        data_rows = [line + "\n" for line in lines[1:]]
        
        return self._distribute_rows(prefix, header_row, suffix, data_rows, total_len, global_start, global_end)

    def _distribute_rows(self, head_wrap: str, header_row: str, tail_wrap: str, data_rows: List[str], total_len: int, global_start: int, global_end: int) -> Tuple[List[str], List[Tuple[int, int]], str]:
        """核心分配逻辑：动态贪心切分。计算绝对坐标并执行滑动窗口式的首尾相连组装"""
        curr_idx = global_start
        
        head_wrap_start = curr_idx
        curr_idx += len(head_wrap)
        curr_idx += len(header_row)
        
        data_rows_with_coords = []
        for row in data_rows:
            r_start = curr_idx
            curr_idx += len(row)
            r_end = curr_idx - 1
            if self.length_function(row) > self.chunk_size:
                logger.warning("检测到单行表格超过 chunk_size，按字符坐标安全拆分")
                for chunk_text, chunk_start, chunk_end in self._split_long_span_text(row, r_start):
                    data_rows_with_coords.append((chunk_start, chunk_end, chunk_text))
            else:
                data_rows_with_coords.append((r_start, r_end, row))
        
        row_groups = []
        if total_len <= 1.5 * self.chunk_size:
            mid = len(data_rows_with_coords) // 2
            row_groups.append(data_rows_with_coords[:mid])
            row_groups.append(data_rows_with_coords[mid:])
        else:
            current_group = []
            current_len = self.length_function(head_wrap + header_row)
            remaining_rows_len = sum(self.length_function(r[2]) for r in data_rows_with_coords)
            
            for i, item in enumerate(data_rows_with_coords):
                r_start, r_end, row = item
                row_len = self.length_function(row)
                if remaining_rows_len <= 1.5 * self.chunk_size:
                    remaining_items = data_rows_with_coords[i:]
                    mid = len(remaining_items) // 2
                    
                    if current_group:
                        row_groups.append(current_group)
                    
                    row_groups.append(remaining_items[:mid])
                    row_groups.append(remaining_items[mid:])
                    
                    current_group = []
                    break 
                
                if current_len + row_len > self.chunk_size and current_group:
                    row_groups.append(current_group)
                    current_group = [item]
                    prev_last_row_text = row_groups[-1][-1][2]
                    current_len = self.length_function(prev_last_row_text) + row_len
                else:
                    current_group.append(item)
                    current_len += row_len
                remaining_rows_len -= row_len
                    
            if current_group:
                row_groups.append(current_group)

        row_groups = [group for group in row_groups if group]

        chunks = []
        coords = []
        note = header_row
        
        for i, group in enumerate(row_groups):
            if not group: 
                continue
                
            if i == 0:
                chunk_text = head_wrap + header_row + "".join(item[2] for item in group)
                chunk_start = head_wrap_start
                chunk_end = group[-1][1]
                    
            elif i == len(row_groups) - 1:
                prev_last_item = row_groups[i-1][-1]
                chunk_text = prev_last_item[2] + "".join(item[2] for item in group) + tail_wrap
                
                chunk_start = prev_last_item[0]
                chunk_end = global_end
                
            else:
                prev_last_item = row_groups[i-1][-1]
                chunk_text = prev_last_item[2] + "".join(item[2] for item in group)
                
                chunk_start = prev_last_item[0]
                chunk_end = group[-1][1]
                
            chunks.append(chunk_text)
            coords.append((chunk_start, chunk_end))

        return chunks, coords, note

    def chunk(self, docs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        主入口函数
        """
        doc_info = docs.get("document_info", {})
        parsed_blocks = self._ensure_block_coordinates(docs.get("parsed_blocks", []))

        self._base_meta = {
            "doc_source": doc_info.get("doc_source", "") if doc_info.get("doc_source", "") != "未知机构" else "",
            "file_name": doc_info.get("file_name", "")
        }
        
        final_chunks = []
        global_chunk_id = 0
        
        grouped_blocks = groupby(parsed_blocks, key=lambda b: b.get("block_label"))
        for label, group_iter in grouped_blocks:
            group = list(group_iter)
            if label == "text":
                group_start = group[0].get("global_start", 0)
                group_end = group[-1].get("global_end", 0)
                split_chunks, coords = self._split_large_text_from_blocks(group, group_start, group_end)
                for chunk_text, coord in zip(split_chunks, coords):
                    final_chunks.append(self._format_chunk(
                        content=chunk_text,
                        label="text",
                        chunk_id=global_chunk_id,
                        part_index=0,
                        global_start=coord[0],
                        global_end=coord[1],
                        note=""
                        ))
                    global_chunk_id += 1
            elif label == "table&chart":
                for block in group:
                    raw_content = block.get("block_content", "")
                    b_start = block.get("global_start", 0)
                    b_end = block.get("global_end", 0)
                    split_chunks, coords, note = self._split_table_and_chart(raw_content, b_start, b_end)
                    for index, (chunk_text, coord) in enumerate(zip(split_chunks, coords)):
                        final_chunks.append(self._format_chunk(
                            content=chunk_text, 
                            label="table&chart", 
                            chunk_id=global_chunk_id, 
                            part_index=index + 1,
                            global_start=coord[0],
                            global_end=coord[1],
                            note=note
                        ))
                    global_chunk_id += 1
        return final_chunks

    def _format_chunk(self, content: str, label: str, chunk_id: int, part_index: int, global_start: int, global_end: int, note: str) -> Dict:
        """组装最终字典，保证每个 chunk 都有完全一致的 metadata 结构"""
        return {
            "metadata": {
                **self._base_meta, 
                "block_id": chunk_id,
                "part_index": part_index,
                "label": label,
                "global_start": global_start,
                "global_end": global_end,
                "note": note
            },
            "content": content
        }

    def _split_large_text(self, text: str, global_start: int, global_end: int) -> List[str]:
        """由子类实现：负责将一段超长的 string 切成不超过 chunk_size 的 List[str]，且 List 元素之间应当包含 overlap"""
        raise NotImplementedError


class RecursiveChineseBlockSplitter(BaseBlockSplitter):
    _SEPARATORS = [
        re.compile(r'(\n{2,})'),
        re.compile(r'(\n)'),
        re.compile(r'([;；.!?。！？\?]["’”」』]{0,2})'),
        re.compile(r'([,，]["’”」』]{0,2})')
    ]
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        spm_model_path: str = Settings.kb_settings.TOKENIZER_FILE
    ):
        super().__init__(chunk_size, chunk_overlap, spm_model_path)

    def _split_large_text_from_blocks(self, blocks: List[Dict], global_start: int, global_end: int) -> Tuple[List[str], List[Tuple[int, int]]]:
        raw_text = "".join(b.get("block_content", "") for b in blocks)
        if not raw_text.strip():
            return [], []
        fine_splits = self._recursive_split_with_rel_pos(raw_text, 0, len(raw_text)-1, self._SEPARATORS)
        has_pre_overlapped_splits = any(
            fine_splits[index][1] <= fine_splits[index - 1][2]
            for index in range(1, len(fine_splits))
        )
        merged = fine_splits if has_pre_overlapped_splits else self._merge_sentences_with_overlap_and_rel_pos(fine_splits, raw_text)
        split_chunks = []
        coords = []
        for chunk_text, rel_start, rel_end in merged:
            split_chunks.append(chunk_text)
            global_start_coord = rel_start + global_start
            global_end_coord = rel_end + global_start
            coords.append((global_start_coord, global_end_coord))
        
        return split_chunks, coords

    def _recursive_split_with_rel_pos(self, text: str, rel_start: int, rel_end: int, separators: List[re.Pattern]) -> List[Tuple[str, int, int]]:
        if self.length_function(text) <= self.chunk_size:
            return [(text, rel_start, rel_end)]
        if not separators:
            return self._hard_split_with_rel_pos(text, rel_start, rel_end)
        
        separator = separators[0]
        new_separators = separators[1:]
        if not separator.search(text):
            return self._recursive_split_with_rel_pos(text, rel_start, rel_end, new_separators)
        
        parts = separator.split(text)
        splits = []
        current_rel_pos = rel_start
        
        for i in range(0, len(parts), 2):
            part = parts[i]
            sep = parts[i+1] if i + 1 < len(parts) else ""
            combined = part + sep
            if combined:
                combined_len = len(combined)
                combined_rel_end = current_rel_pos + combined_len - 1
                splits.append((combined, current_rel_pos, combined_rel_end))
                current_rel_pos = combined_rel_end + 1
        
        fine_splits = []
        for s_text, s_rel_start, s_rel_end in splits:
            if not s_text:
                continue
            if self.length_function(s_text) <= self.chunk_size:
                fine_splits.append((s_text, s_rel_start, s_rel_end))
            else:
                fine_splits.extend(self._recursive_split_with_rel_pos(s_text, s_rel_start, s_rel_end, new_separators))
        
        return fine_splits

    def _hard_split_with_rel_pos(self, text: str, rel_start: int, rel_end: int) -> List[Tuple[str, int, int]]:
        return self._split_long_span_text(text, rel_start)

    def _merge_sentences_with_overlap_and_rel_pos(self, sentences_with_rel_pos: List[Tuple[str, int, int]], raw_text: str) -> List[Tuple[str, int, int]]:
        chunks = []
        current_chunk = []
        current_len = 0

        for s_text, s_rel_start, s_rel_end in sentences_with_rel_pos:
            sentence_len = self.length_function(s_text)
            
            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunk_text = "".join([t for t, _, _, _ in current_chunk])
                chunk_rel_start = current_chunk[0][1]
                chunk_rel_end = current_chunk[-1][2]
                chunks.append((chunk_text, chunk_rel_start, chunk_rel_end))
                overlap = []
                overlap_len = 0
                for t, start, end, l in reversed(current_chunk):
                    if overlap_len + l <= self.chunk_overlap:
                        overlap.append((t, start, end, l))
                        overlap_len += l
                    else:
                        break
                
                overlap.reverse()
                if not overlap:
                    tail_start_in_chunk = self._find_overlap_start_by_length(chunk_text, 0, len(chunk_text))
                    tail_text = chunk_text[tail_start_in_chunk:]
                    tail_len = self.length_function(tail_text)
                    tail_rel_start = chunk_rel_start + tail_start_in_chunk
                    tail_rel_end = chunk_rel_end
                    current_chunk = [(tail_text, tail_rel_start, tail_rel_end, tail_len)]
                    current_len = tail_len
                else:
                    current_chunk = overlap
                    current_len = overlap_len
            
            current_chunk.append((s_text, s_rel_start, s_rel_end, sentence_len))
            current_len += sentence_len
        if current_chunk:
            chunk_text = "".join([t for t, _, _, _ in current_chunk])
            chunk_rel_start = current_chunk[0][1]
            chunk_rel_end = current_chunk[-1][2]
            chunks.append((chunk_text, chunk_rel_start, chunk_rel_end))
        
        return chunks
    
    
def make_text_splitter(
    splitter_name: str,
    chunk_size: int,
    chunk_overlap: int,
    **kwargs
):
    """
    Text Splitter 的工厂函数。
    根据 splitter_name 返回对应的 Splitter 实例。
    """
    SPLITTER_REGISTRY = {
        "RecursiveChineseBlockSplitter": RecursiveChineseBlockSplitter,
        # 未来扩展示例：
        # "SimpleEnglishSplitter": SimpleEnglishSplitter, 
        # "CodeBlockSplitter": CodeBlockSplitter,
    }

    SplitterClass = SPLITTER_REGISTRY.get(splitter_name)
    
    if not SplitterClass:
        raise ValueError(f"未支持的 text_splitter_name: '{splitter_name}'。当前支持: {list(SPLITTER_REGISTRY.keys())}")

    return SplitterClass(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        **kwargs
    )


class KnowledgeFile:
    def __init__(
        self,
        filename: str,
        knowledge_base_name: str,
    ):
        """
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        """
        self.kb_name = knowledge_base_name
        self.filename = str(Path(filename).as_posix())
        self.ext = os.path.splitext(filename)[-1].lower()
        file_stem = os.path.splitext(os.path.basename(self.filename))[0]
        self.filepath = get_file_path(knowledge_base_name, self.filename)
        self.jsonPath = get_json_path(knowledge_base_name, file_stem + ".json")
        self.text_splitter_name = Settings.kb_settings.TEXT_SPLITTER_NAME
        self.docs = None
        self.splited_docs = None
    
    def file2docs(self, refresh: bool = False) -> Dict:
        """只负责把文件读进内存"""
        if self.docs is None or refresh:
            with open(self.jsonPath, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
        return self.docs

    def docs2texts(
        self,
        docs: Dict[str, Any] = None,
        refresh: bool = False,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
        text_splitter = None,
    ) -> List:
        """只负责切分传入的 docs，或者切分已经缓存的 docs"""
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
            
        if text_splitter is None:
            text_splitter = make_text_splitter(
                splitter_name=self.text_splitter_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        self.splited_docs = text_splitter.chunk(docs)
        return self.splited_docs

    def file2text(
        self,
        refresh: bool = False,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
        text_splitter = None,
    ) -> List:
        """主入口：如果已经有缓存就直接返回，否则执行一键转换"""
        if self.splited_docs is not None and not refresh:
            return self.splited_docs
        
        return self.docs2texts(
            docs=None, 
            refresh=refresh,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            text_splitter=text_splitter
        )

    def file_exist(self):
        return os.path.isfile(self.jsonPath)

    def get_mtime(self):
        return os.path.getmtime(self.jsonPath)

    def get_size(self):
        return os.path.getsize(self.jsonPath)


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = None,
) -> Generator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    """
    if params is None:
        params = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        tasks = {pool.submit(func, **kwargs): kwargs for kwargs in params}

        for future in as_completed(tasks):
            try:
                yield future.result()
            except BaseException as e:
                failed_params = tasks[future]
                logger.exception(f"Error in sub thread with params {failed_params}: {e}")


def files2docs_in_thread_file2docs(
    *, file: KnowledgeFile, **kwargs
) -> Tuple[bool, Tuple[str, str, List[Dict[str, Any]]]]:
    try:
        return True, (file.kb_name, file.filename, file.file2text(**kwargs))
    except Exception as e:
        msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return False, (file.kb_name, file.filename, msg)


def files2docs_in_thread(
    files: List[KnowledgeFile],
    chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
) -> Generator:
    kwargs_list = [
        {
            "file": file,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        for file in files
    ]

    for result in run_in_thread_pool(
        func=files2docs_in_thread_file2docs, 
        params=kwargs_list
    ):
        yield result


def format_reference(kb_name: str, docs: List[Dict], api_base_url: str="") -> List[Dict]:
    '''
    将知识库检索结果格式化为参考文档的格式
    '''
    from Finance_RAG.utils import api_address
    api_base_url = api_base_url or api_address(is_public=True)

    source_documents = []
    for inum, doc in enumerate(docs):
        filename = doc.get("metadata", {}).get("file_name")
        filename_with_suffix = f"{filename}.pdf"
        real_file_path = f"content/{filename_with_suffix}"
        
        parameters = urlencode(
            {
                "knowledge_base_name": kb_name,
                "file_name": real_file_path,  # 👈 这里换成真实路径
            }
        )
        api_base_url = api_base_url.strip(" /")
        url = (
            f"{api_base_url}/knowledge_base/download_doc?" + parameters
        )
        page_content = doc.get("content")
        ref = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{page_content}\n\n"""
        source_documents.append(ref)
    
    return source_documents


if __name__ == "__main__":
    kb_file = KnowledgeFile(
        filename="E:\\LLM\\Data\\Test.md", knowledge_base_name="samples"
    )
    kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    docs = kb_file.file2docs()
    texts = kb_file.docs2texts(docs)
    for text in texts:
        print(text)
