import json
from pathlib import Path

def rename_pdfs_and_record_mapping(base_dir_path: str):
    """
    批量将各个目录下的 PDF 重命名为 类别缩写_编号.pdf 的格式，
    并生成一个 mapping.json 文件记录 新旧文件名 的映射关系。
    """
    base_dir = Path(base_dir_path)
    
    category_abbr = {
        "Aerospace_equipment": "Aerospace_equipment",
        "Battery": "Battery",
        "Electricity": "Electricity",
        "Semiconductor": "Semiconductor"
    }
    
    global_mapping = {}

    for folder_name, abbr in category_abbr.items():
        folder_dir = base_dir / folder_name
        
        if not folder_dir.exists() or not folder_dir.is_dir():
            print(f"警告: 未找到文件夹 {folder_dir}，已跳过。")
            continue
            
        print(f"正在处理文件夹: {folder_name} ...")
        global_mapping[folder_name] = {}
        pdf_files = sorted(folder_dir.glob("*.pdf"))
        for index, pdf_path in enumerate(pdf_files, start=1):
            original_name = pdf_path.name

            if original_name.startswith(f"{abbr}_"):
                continue
                
            new_name = f"{abbr}_{index}.pdf"
            new_path = folder_dir / new_name
            global_mapping[folder_name][new_name] = original_name
            pdf_path.rename(new_path)

    mapping_file = base_dir / "pdf_rename_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(global_mapping, f, ensure_ascii=False, indent=4)
        
    print(f"\n全部处理完成！")
    print(f"映射文件已保存至: {mapping_file}")
    
    return global_mapping

if __name__ == "__main__":
    target_path = "/2022110126/lpr_pjx/LLM-RAG/Finance_RAG/pdfs/"
    mapping_result = rename_pdfs_and_record_mapping(target_path)