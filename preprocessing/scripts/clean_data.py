import json
from pathlib import Path

def validate_sharegpt_format(item: dict) -> bool:
    """复刻 validate.py 的核心验证逻辑"""
    if "conversations" not in item:
        return False
    
    convs = item["conversations"]
    if not isinstance(convs, list) or len(convs) < 3:
        return False
    
    roles = [c.get("from") for c in convs]
    if "system" not in roles or "human" not in roles or "gpt" not in roles:
        return False
    
    # 检查 gpt 回复是否完整包含 <think></think>
    gpt_value = ""
    for conv in convs:
        if conv.get("from") == "gpt":
            gpt_value = conv.get("value", "")
            break
    
    if not gpt_value or "<think>" not in gpt_value or "</think>" not in gpt_value:
        return False
    
    return True

def main():
    input_file = Path("../data/finetune/synthesized/medalpaca_cot.jsonl")
    output_file = Path("../data/finetune/synthesized/medalpaca_cot_cleaned.jsonl")
    
    if not input_file.exists():
        print(f"文件不存在: {input_file}")
        return

    valid_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total_count += 1
            item = json.loads(line)
            if validate_sharegpt_format(item):
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                valid_count += 1
    
    print(f"处理完成！")
    print(f"原始条数: {total_count}")
    print(f"保留条数: {valid_count}")
    print(f"删除条数: {total_count - valid_count}")
    
    # 替换原文件
    import os
    os.replace(output_file, input_file)
    print(f"原文件已更新。")

if __name__ == "__main__":
    main()
