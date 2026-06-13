from paddleocr import PaddleOCRVL
import re
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import json
import fitz
import pathlib
import sys
sys.path.append("..") 
from Finance_RAG.utils import build_logger
from Finance_RAG.settings import Settings

logger = build_logger()

#paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8118

class DocumentDetector:
    @staticmethod
    def pdf_to_ndarrays(pdf_path: str, dpi: int = 150) -> Dict[str, Any]:
        path_obj = pathlib.Path(pdf_path)
        
        if not path_obj.exists():
            return {"is_valid": False, "error": f"文件不存在: {pdf_path}"}
        
        try:
            with open(path_obj, "rb") as f:
                stream = f.read()
                doc = fitz.open(stream=stream, filetype="pdf")
                
                images_ndarray = []
                for page in doc:
                    pix = page.get_pixmap(dpi=dpi)
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img_array = img_array[:, :, :3]
                    images_ndarray.append(img_array)
                
                doc.close()
                
            return {
                "is_valid": True, 
                "page_count": len(images_ndarray), 
                "file_name": path_obj.stem,
                "images": images_ndarray
            }
        except Exception as e:
            print(f"PDF 转换失败: {e}") 
            return {"is_valid": False, "error": str(e)}


class VisionLayoutParser:
    def __init__(self):
        self.pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            use_queues=True,
            vl_rec_backend="vllm-server",
            vl_rec_server_url = Settings.basic_settings.paddle_model_url or "http://127.0.0.1:8118/v1"
        )

    def parse_and_dump(self, ndarray_list: List[np.ndarray]):
        raw_results = self.pipeline.predict(
            input=ndarray_list,
            use_chart_recognition=True,
            use_ocr_for_image_block=True
        )
        
        restructured_results = self.pipeline.restructure_pages(
            res_list=list(raw_results), 
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=False
        )
        return restructured_results


class StructuredDocumentBuilder:
    def __init__(self):
        self.detector = DocumentDetector()
        self.parser = VisionLayoutParser()
        self.ignore_labels = {"header_image", "header", "footer_image", "footer", "number", "aside_text","doc_title"}
        self.noise_patterns = [
            re.compile(r"^[\u4e00-\u9fa5]{0,4}(?:分析师|研究员|联系人)[：:\s]+[\u4e00-\u9fa5]{2,4}"),
            re.compile(r"(?:证书|登记|执业|从业|SAC|SFC).*?(?:编号)?[：:\s]+[A-Za-z0-9]{5,}"),
            re.compile(r"(?:电话|传真|手机|联系方式|热线|客服)[：:]?\s*[\d\-\+()（）]{8,}|\b\d{3,4}-\d{7,8}\b"),
            re.compile(r"(?:邮箱|E-mail|Email)[：:\s]*|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.IGNORECASE),
            re.compile(r"邮(政)?编(码)?[：:\s]*\d{6}"),
            re.compile(r"(?:网址|网站|Web)[：:\s]*|https?://[\w\-]+(\.[\w\-]+)+|www\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
            re.compile(r"(?:地址[:：])?[\u4e00-\u9fa5\d]+(?:省|市|自治区|县|区|旗)?[\u4e00-\u9fa5\d]+(?:路|街|大道|巷|弄|号|院|小区|花园|广场|大厦|中心|楼|座|层|室)+.*")
        ]
        symbols = (
            r"☑■□▢▣▤▥▦▧▨▩▪▫▬▭▮▯▰▱►▼◄▲"
            r"○●◎◇◆□"
            r"★☆✡✦✧✩✪✫✬✭✮✯"
            r"①②③④⑤⑥⑦⑧⑨⑩"
            r"➊➋➌➍➎➏➐➑➒➓"
            r"➔➕➖➗"
            r"◦•∙※‣⁃"
            r"\u00A0\u3000\u2000-\u200F\u2028-\u202F"
        )
        self.symbol_pattern = re.compile(f"[{symbols}]")
        self.zh_space_pattern = re.compile(r'(?<=[\u4e00-\u9fa5])\s+|\s+(?=[\u4e00-\u9fa5])')
        self.disclaimer_pattern = re.compile(r"免责|声明|评级|分析师|简介|提示|报告汇总|介绍")
        self.end_punctuations = ('。', '！', '？', '.', '!', '?', '；', ';')
        self.table_top_clean_re = re.compile(r'</table>\s*$', flags=re.IGNORECASE)
        self.table_bottom_clean_tr_re = re.compile(r'^.*?<table[^>]*>.*?</tr>', flags=re.IGNORECASE | re.DOTALL)
        self.table_bottom_clean_fallback_re = re.compile(r'^.*?<table[^>]*>', flags=re.IGNORECASE)
        self.title_pattern = r'^(第[一二三四五六七八九十百]+[章节条篇]|[\d一二三四五六七八九十]+\s*[、\.]|[\(（][\d一二三四五六七八九十]+[\)）]|\d+\.\d+)'
        
    def parse_pdf(self, pdf_path: str, dpi: int = 150, save_json: bool = False) -> Dict[str, Any]:
        try:
            doc_info = self.detector.pdf_to_ndarrays(pdf_path, dpi=dpi)
            if not doc_info["is_valid"]:
                logger.error(f"PDF解析失败: {pdf_path}, 错误: {doc_info.get('error')}")
                return {}

            ndarray_images = doc_info["images"]
            file_name = doc_info["file_name"]
            raw_results = self.parser.parse_and_dump(ndarray_images)

            final_data = self.build(raw_results, file_name)
            if not final_data:
                return {}
        except Exception as e:
            logger.exception(f"PDF处理/版面分析失败: {pdf_path}")
            return {}

        if save_json:
            pdf_path_obj = Path(pdf_path)
            parent_dir = pdf_path_obj.parent
            json_path = parent_dir.parent / "raw_resolve" / f"{pdf_path_obj.stem}.json"

            if json_path.exists():
                logger.info(f"文件已存在，跳过保存：{json_path}")
                return final_data
            
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)
            logger.info(f"已保存结果: {json_path}")

        return final_data
        
        
    def build(self, raw_pages: List[Dict[str, Any]], file_name: str) -> Dict[str, Any]:
        """主干 Pipeline"""
        if not raw_pages:
            return {}
        global_meta = self._extract_global_meta(raw_pages)
        global_meta["file_name"] = file_name
        cleaned_blocks = self._clean_and_preserve_pages(raw_pages)
        grouped_blocks = self._reorder_and_group(cleaned_blocks)
        
        return {
            "document_info": global_meta,
            "parsed_blocks": grouped_blocks
        }

    def _extract_global_meta(self, raw_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取全局信息"""
        meta = {
            "doc_source": "未知机构"
        }
        first_page_blocks = raw_pages[0].get("parsing_res_list", [])
        found_source = False

        for block in first_page_blocks:
            label = block.label
            content = block.content
            
            if not found_source and label == "header_image":
                meta["doc_source"] = content.split("\n")[0].strip()
                found_source = True
                break

        return meta

    def _clean_and_preserve_pages(self, raw_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """初步清洗噪声，精简字段，并在此时将 Block 对象彻底转化为纯字典"""
        cleaned_blocks = []
        total_pages = len(raw_pages)
        in_toc_section = False
        in_useless_section = False

        for i, page in enumerate(raw_pages):
            page_idx = page.get('page_index')
            blocks = page.get('parsing_res_list', [])
            if page_idx is None:
                page_idx = i
            is_tail_page = (total_pages - i) <= 3

            for block in blocks:
                label = block.label
                content = block.content
                bbox = block.bbox
                if label in self.ignore_labels or not content:
                    continue
                if label == "paragraph_title":
                    if content.endswith("目录"):
                        in_toc_section = True
                    else:
                        in_toc_section = False
                if in_toc_section:
                    continue

                if is_tail_page:
                    if label == "paragraph_title":
                        if self.disclaimer_pattern.search(content):
                            in_useless_section = True
                        else:
                            in_useless_section = False
                    if in_useless_section:
                        continue
                if any(pattern.search(content) for pattern in self.noise_patterns):
                    continue
                clean_content = self.symbol_pattern.sub("", content)
                if label in ["table", "chart"]:
                    clean_content = re.sub(r'[ \t\f\v]+', ' ', clean_content)
                    clean_content = re.sub(r'\n+', '\n', clean_content).strip()
                else:
                    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                    clean_content = self.zh_space_pattern.sub('', clean_content)

                slim_block = {
                    "block_label": label,
                    "block_content": clean_content,
                    "block_bbox": bbox,
                    "source_page": page_idx
                }
                cleaned_blocks.append(slim_block)

        return cleaned_blocks
    
    def _reorder_and_group(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于坐标校验与打标签法，处理图表组装及跨页合并。
        to_delete 标签含义：
        0: 初始状态
        1: 已被融合或无用，最后需要被删除
        2: 独立保留的有效块 (文本或融合后的 composite_chart)
        """
        end_punctuations = ('。', '！', '？', '.', '!', '?', '；', ';')

        grouped_result = []
        num_blocks = len(blocks)

        for i, block in enumerate(blocks):
            if block.get("to_delete", 0) == 1:
                continue

            label = block.get("block_label")
            current_page = block.get("source_page")
            current_bbox = block.get("block_bbox", [0, 0, 0, 0])
            current_content = block.get("block_content", "")

            if label == "table" or label == "chart":
                block["to_delete"] = 2
                block["block_label"] = "table&chart"

                for j in range(i - 1, max(-1, i - 4), -1):
                    prev_block = blocks[j]
                    if prev_block.get("block_label") == "figure_title":
                        prev_page = prev_block.get("source_page")
                        prev_bbox = prev_block.get("block_bbox", [0, 0, 0, 0])

                        if prev_page == current_page and prev_bbox[3] > current_bbox[1]:
                            continue 

                        block["block_content"] = f"title:{prev_block.get('block_content', '')} 内容：{current_content}"
                        prev_block["to_delete"] = 1
                        break

                search_page = current_page
                search_bottom = current_bbox[3]

                for j in range(i + 1, min(num_blocks, i + 4)):
                    next_block = blocks[j]
                    next_label = next_block.get("block_label")
                    next_content = next_block.get("block_content", "")
                    next_page = next_block.get("source_page")
                    next_bbox = next_block.get("block_bbox", [0, 0, 0, 0])

                    if next_page > search_page and next_label == "figure_title":
                        next_block["to_delete"] = 1
                        continue

                    if next_label == "table":
                        if next_page == search_page and next_bbox[1] < search_bottom:
                            break 

                        block["block_content"] = self._merge_cross_page_tables(block['block_content'], next_content)
                        next_block["to_delete"] = 1
                        search_page = next_page
                        search_bottom = next_bbox[3]
                        continue

                    if next_label == "vision_footnote" or (next_label == "text" and "资料来源" in next_content):
                        if next_page == search_page and next_bbox[1] < search_bottom:
                            if search_bottom - next_bbox[1] > 20: 
                                continue

                        block["block_content"] = f"{block['block_content']} 脚注{next_content}"
                        next_block["to_delete"] = 1
                        break

                    if next_label in ["text", "paragraph_title"] and "资料来源" not in next_content:
                        break

                grouped_result.append(block)

            else:
                block["to_delete"] = 2
                if label in ["text", "paragraph_title"]:
                    curr_idx = i
                    content_parts = [block["block_content"]]

                    while curr_idx + 1 < num_blocks:
                        next_block = blocks[curr_idx + 1]
                        if next_block.get("to_delete") == 1:
                            break
                        curr_page = blocks[curr_idx].get("source_page")
                        next_page = next_block.get("source_page")
                        next_label = next_block.get("block_label")
                        next_content = next_block.get("block_content", "")
                        if next_page > curr_page:
                            if label == "text" and next_label == "text":
                                if not content_parts[-1].endswith(end_punctuations):
                                    content_parts.append(next_content)
                                    next_block["to_delete"] = 1
                                    curr_idx += 1
                                    continue
                            elif label == "paragraph_title" and next_label == "paragraph_title":
                                if not re.match(self.title_pattern, next_content.strip()):
                                    content_parts.append(next_content)
                                    next_block["to_delete"] = 1
                                    curr_idx += 1
                                    continue
                                else:
                                    break
                        break
                    block["block_content"] = "".join(content_parts)

                grouped_result.append(block)

        final_result = []
        current_paragraph_title = ""
        current_global_idx = 1

        for b in grouped_result:
            if b.get("to_delete") == 1:
                continue
            b.pop("to_delete", None)

            label = b.get("block_label")
            if label == "paragraph_title":
                current_paragraph_title = b.get("block_content", "")
                continue
            if current_paragraph_title:
                b["paragraph_title"] = current_paragraph_title
            content = b.get("block_content", "")
            content_len = len(content)
            
            if content_len > 0:
                b["global_start"] = current_global_idx
                b["global_end"] = current_global_idx + content_len - 1 
                current_global_idx = b["global_end"] + 1
            final_result.append(b)
        return final_result



    def _merge_cross_page_tables(self, table_top: str, table_bottom: str) -> str:
        """辅助函数：处理表格 HTML 的跨页拼接，剥离下半截的第一行 <tr> 表头"""
        top_clean = self.table_top_clean_re.sub('', table_top)
        bottom_clean = self.table_bottom_clean_tr_re.sub('', table_bottom, count=1)
        if bottom_clean == table_bottom:
            bottom_clean = self.table_bottom_clean_fallback_re.sub('', table_bottom)
        return top_clean + bottom_clean


def batch_process_pdfs(input_dir: str):
    root_path = Path(input_dir)
    logger.info("初始化文档解析器...")
    builder = StructuredDocumentBuilder()
    logger.info("初始化完成！")

    pdf_files = list(root_path.rglob("*.pdf"))
    logger.info(f"共发现 {len(pdf_files)} 个 PDF 文件准备处理。")

    for pdf_path in pdf_files:
        logger.info(f"开始处理: {pdf_path.name}")
        
        final_structured_data = builder.parse_pdf(str(pdf_path), save_json=True)

        if not final_structured_data:
            logger.error(f"处理失败: {pdf_path.name}")
            continue

        logger.info(f"处理完成: {pdf_path.name}")


if __name__ == "__main__":
    INPUT_ROOT = "/2022110126/lpr_pjx/LLM-RAG/Finance_RAG/Data/knowledge_base/Finance_faiss256_flat/content"
    batch_process_pdfs(INPUT_ROOT)