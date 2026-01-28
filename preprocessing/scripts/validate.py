"""
Validation script - 验证生成数据质量
"""
import json
import sys
from pathlib import Path
import random


def validate_sharegpt_format(item: dict) -> tuple[bool, str]:
    """验证ShareGPT格式"""
    if "conversations" not in item:
        return False, "缺少conversations字段"
    
    convs = item["conversations"]
    if not isinstance(convs, list) or len(convs) < 3:
        return False, "conversations格式错误或条目不足"
    
    # 检查必要的角色
    roles = [c.get("from") for c in convs]
    if "system" not in roles or "human" not in roles or "gpt" not in roles:
        return False, "缺少必要的角色(system/human/gpt)"
    
    # 检查gpt回复是否完整包含 <think></think> 标签
    gpt_value = ""
    for conv in convs:
        if conv.get("from") == "gpt":
            gpt_value = conv.get("value", "")
            break
    
    if gpt_value:
        if "<think>" not in gpt_value or "</think>" not in gpt_value:
            return False, "gpt回复缺少完整的 <think></think> 标签"
    
    return True, "OK"


def load_data(file_path: Path) -> list[dict]:
    """支持加载 JSON 和 JSONL 格式"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix == '.jsonl':
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else:
            data = json.load(f)
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="验证合成数据质量")
    parser.add_argument("--file", required=True, help="数据文件路径")
    parser.add_argument("--sample", type=int, default=5, help="随机抽样数量")
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[错误] 文件不存在: {args.file}")
        return

    # 加载数据
    try:
        data = load_data(file_path)
    except Exception as e:
        print(f"[错误] 加载文件失败: {str(e)}")
        return

    print(f"数据总量: {len(data)}")
    
    # 统计信息
    stats = {
        "exam": 0,
        "dialogue": 0,
        "total_tokens": 0,
        "total_calls": 0,
        "format_ok": 0,
        "format_err": 0
    }
    
    format_errors = []
    for idx, item in enumerate(data):
        # 统计元数据
        meta = item.get("metadata", {})
        task_type = meta.get("task_type", "unknown")
        if task_type in stats:
            stats[task_type] += 1
        
        stats["total_tokens"] += meta.get("tokens_used", 0)
        stats["total_calls"] += meta.get("api_calls", 0)

        # 验证格式
        is_valid, reason = validate_sharegpt_format(item)
        if is_valid:
            stats["format_ok"] += 1
        else:
            stats["format_err"] += 1
            format_errors.append({"index": idx, "reason": reason})
    
    print(f"--- 任务分布 ---")
    print(f"Exam: {stats['exam']} | Dialogue: {stats['dialogue']}")
    print(f"--- 资源消耗 ---")
    print(f"总 Tokens: {stats['total_tokens']} | 总 API 调用: {stats['total_calls']}")
    print(f"--- 格式验证 ---")
    print(f"格式正确: {stats['format_ok']}")
    print(f"格式错误: {stats['format_err']}")
    
    if format_errors:
        print("\n[错误] 格式错误详情（前5条）:")
        for error in format_errors[:5]:
            print(f"  - 索引 {error['index']}: {error['reason']}")
    
    # 随机抽样展示
    if args.sample > 0 and len(data) > 0:
        print(f"\n随机抽样 {min(args.sample, len(data))} 条展示:")
        samples = random.sample(data, min(args.sample, len(data)))
        
        for idx, item in enumerate(samples, 1):
            print(f"\n{'='*20} 样本 {idx} {'='*20}")
            meta = item.get("metadata", {})
            print(f"Task ID: {item.get('id', 'N/A')} | Type: {meta.get('task_type', 'N/A')}")
            
            for conv in item.get("conversations", []):
                role = conv.get("from", "unknown")
                value = conv.get("value", "")
                # 只打印前150个字符和最后100个字符，方便观察 <think> 结构
                if len(value) > 250:
                    display_val = value[:150] + "\n... [中间省略] ...\n" + value[-100:]
                else:
                    display_val = value
                print(f"\n[{role.upper()}]:")
                print(display_val)


if __name__ == "__main__":
    main()
