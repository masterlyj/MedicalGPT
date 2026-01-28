"""
Main synthesis script - 数据合成主脚本
"""
import asyncio
import json
from pathlib import Path
import sys
import aiofiles

from src.state import AgentState
from src.workflow import create_workflow
from src.api_router import APIRouter
from src.logger import SynthesisLogger


def normalize_input(task_data: dict, task_type: str) -> dict:
    """
    统一输入格式适配器
    根据命令行指定的 task_type，从不同格式中提取内容
    """
    import hashlib
    
    # 1. 提取核心内容用于生成 ID 和处理
    content = ""
    answer = task_data.get("answer")
    
    if "conversations" in task_data:
        for turn in task_data["conversations"]:
            if turn.get("from") in ["human", "user"]:
                content = turn.get("value", "")
                break
    elif "instruction" in task_data:
        instruction = task_data.get("instruction", "")
        input_text = task_data.get("input", "")
        content = f"{instruction}\n{input_text}".strip()
        if not answer:
            answer = task_data.get("output")
    else:
        content = task_data.get("content") or task_data.get("text") or ""

    # 2. 确定性 ID 生成逻辑
    # 如果原始数据有 id，优先使用；否则根据 content 生成 MD5 哈希
    if "id" in task_data and task_data["id"]:
        task_id = str(task_data["id"])
    else:
        # 使用内容哈希确保：相同内容 -> 相同ID；不同内容 -> 不同ID
        # 用于增量更新（--resume）
        hash_obj = hashlib.md5(content.encode('utf-8'))
        task_id = f"gen_{hash_obj.hexdigest()[:16]}"
        
    return {
        "id": task_id,
        "type": task_type,
        "content": content,
        "answer": answer
    }


async def process_single_task(workflow, task_data: dict, task_type: str) -> dict:
    """处理单个任务"""
    # 格式标准化
    normalized = normalize_input(task_data, task_type)
    
    if not normalized["content"]:
        raise ValueError(f"Task {normalized['id']} has empty content")

    state = AgentState(
        task_id=normalized["id"],
        task_type=normalized["type"],
        raw_content=normalized["content"],
        answer=normalized["answer"]
    )
    
    # 运行工作流
    result = await workflow.ainvoke(state)
    
    # 统一转换为 ShareGPT 格式输出
    output = {
        "id": result["task_id"],
        "conversations": [
            {
                "from": "system",
                "value": result["system_prompt"] if result["system_prompt"] else "你是一名专业的医学 AI 助手。"
            },
            {
                "from": "human",
                "value": result["raw_content"]
            },
            {
                "from": "gpt",
                "value": result["final_response"] if result["final_response"] else result["draft_response"]
            }
        ],
        "metadata": {
            "task_type": result["task_type"],
            "tokens_used": result["tokens_used"],
            "api_calls": result["api_calls"],
            "revision_count": result["revision_count"],
            "errors": result["errors"],
            "original_data": task_data
        }
    }
    
    return output


async def process_batch(workflow, tasks: list, task_type: str, logger: SynthesisLogger, output_file: Path, max_workers: int = 10, append: bool = False):
    """批量处理任务并实时写入文件"""
    semaphore = asyncio.Semaphore(max_workers)
    
    # 根据是否是增量更新选择打开模式
    mode = 'a' if append else 'w'
    
    async with aiofiles.open(output_file, mode=mode, encoding='utf-8') as f:
        async def process_with_semaphore(task):
            async with semaphore:
                try:
                    result = await process_single_task(workflow, task, task_type)
                    
                    # 只有没有错误的数据才存入主文件
                    if not result["metadata"]["errors"]:
                        await f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        await f.flush() # 确保实时写入
                        logger.log_success()
                        return True
                    else:
                        # 有错误的数据记录到错误日志
                        error_msg = "; ".join(result["metadata"]["errors"])
                        logger.log_error(result["id"], error_msg)
                        return False
                except Exception as e:
                    logger.log_error(task.get("id", "unknown"), str(e)[:200] + "..." if len(str(e)) > 200 else str(e))
                    return False
        
        results = await asyncio.gather(*[process_with_semaphore(task) for task in tasks])
        return sum(1 for r in results if r)


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗数据集合成")
    parser.add_argument("--input", required=True, help="输入文件路径 (.json 或 .jsonl)")
    parser.add_argument("--output", required=True, help="输出JSONL文件路径")
    parser.add_argument("--type", required=True, choices=["exam", "dialogue"], help="任务类型：exam 或 dialogue")
    parser.add_argument("--workers", type=int, default=10, help="并发数")
    parser.add_argument("--resume", action="store_true", help="是否开启增量更新（跳过已存在的ID）")
    args = parser.parse_args()
    
    # 确保输出路径后缀正确
    output_path = Path(args.output)
    if output_path.suffix != '.jsonl':
        output_path = output_path.with_suffix('.jsonl')
    
    # 1. 加载已处理的 ID (用于增量更新)
    processed_ids = set()
    if args.resume and output_path.exists():
        print(f"正在检查已处理的数据...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        processed_ids.add(data["id"])
                except:
                    continue
        print(f"已跳过 {len(processed_ids)} 条已处理的数据")

    # 2. 加载输入数据并预处理 ID
    input_path = Path(args.input)
    raw_tasks = []
    if input_path.suffix == '.jsonl':
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_tasks = [json.loads(line) for line in f if line.strip()]
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_tasks = json.load(f)
    
    # --- 全局查重分析 ---
    input_id_map = {} # task_id -> first_index
    global_duplicates = []
    for i, t in enumerate(raw_tasks):
        normalized = normalize_input(t, args.type)
        task_id = str(normalized["id"])
        if task_id in input_id_map:
            content_snippet = normalized["content"][:30].replace('\n', ' ') + "..."
            global_duplicates.append((i + 1, input_id_map[task_id] + 1, task_id, content_snippet))
        else:
            input_id_map[task_id] = i
    
    if global_duplicates:
        print(f"\n[注意] 输入文件中发现 {len(global_duplicates)} 条重复行：")
        for curr_line, first_line, tid, content in global_duplicates:
            print(f"  - 第 {curr_line} 行与第 {first_line} 行重复 | ID: {tid} | 内容: {content}")
        print("-" * 50)
    # ------------------

    # 3. 预先计算待处理任务 (过滤已存在的 ID)
    tasks_to_process = []
    skipped_count = 0
    seen_in_batch = set()
    
    for t in raw_tasks:
        normalized = normalize_input(t, args.type)
        task_id = str(normalized["id"])
        
        # 已经在输出文件中，跳过
        if task_id in processed_ids:
            skipped_count += 1
            continue
            
        # 在本次处理中已出现过（针对上面 global_duplicates 没处理完的情况），跳过
        if task_id in seen_in_batch:
            continue
            
        seen_in_batch.add(task_id)
        tasks_to_process.append(t)
    
    if not tasks_to_process:
        print("所有数据已处理完成，无需继续。")
        return

    # 4. 初始化
    api_router = APIRouter()
    try:
        workflow = create_workflow(api_router)
        
        # 错误日志路径
        error_log_path = output_path.parent / f"{output_path.stem}_errors.jsonl"
        logger = SynthesisLogger(
            total_tasks=len(tasks_to_process),
            log_file=str(error_log_path)
        )
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"任务类型: {args.type}")
        print(f"原始数据: {len(raw_tasks)} 条")
        print(f"已跳过(已存在): {skipped_count} 条")
        print(f"剩余处理: {len(tasks_to_process)} 条")
        
        # 4. 批量处理并实时写入
        success_count = await process_batch(workflow, tasks_to_process, args.type, logger, output_path, args.workers, append=args.resume)
        
        # 5. 统计
        logger.summary()
        print(f"\n结果已实时保存至: {output_path}")
        if success_count < len(tasks_to_process):
            print(f"失败任务已记录至: {error_log_path}")
        print(f"本次成功率: {success_count}/{len(tasks_to_process)} ({success_count/len(tasks_to_process)*100:.1f}%)")
    finally:
        await api_router.close()


if __name__ == "__main__":
    asyncio.run(main())
