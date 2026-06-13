import json
import os
from pathlib import Path

def patch_json_coordinates(json_dir: str):
    root_path = Path(json_dir)
    json_files = list(root_path.rglob("*.json"))
    
    if not json_files:
        print(f"在目录{json_dir}下没有找到 JSON 文件！")
        return

    print(f"找到{len(json_files)}个JSON 文件，开始补充坐标+删除doc_title...\n")
    
    success_count = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "document_info" in data:
                data["document_info"].pop("doc_title", None)
            original_blocks = data.get("parsed_blocks", [])
            if original_blocks:
                current_global_idx = 1
                for b in original_blocks:
                    content = b.get("block_content", "")
                    content_len = len(content)
                    if content_len > 0:
                        b["global_start"] = current_global_idx
                        b["global_end"] = current_global_idx + content_len - 1 
                        current_global_idx = b["global_end"] + 1

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            print(f"[成功] 已处理: {file_path.name}（坐标已补充 + doc_title 已删除）")
            success_count += 1
            
        except Exception as e:
            print(f"[失败] 处理 {file_path.name} 时出错: {e}")

    print(f"\n全部处理完毕！共成功更新 {success_count} 个文件。")


if __name__ == "__main__":
    JSON_DIR = "./Data/knowledge_base/Finance/raw_resolve" 
    
    patch_json_coordinates(JSON_DIR)