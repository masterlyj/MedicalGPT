# -*- coding: utf-8 -*-
"""
@description: C-Eval格式数据集评测脚本（适用于Qwen2.5模型）
支持有答案的C-Eval格式数据评测
"""
import torch
import json
import re
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from loguru import logger
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


def load_ceval_data(file_path):
    """加载C-Eval格式数据"""
    logger.info(f"Loading C-Eval data from {file_path}")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []


def format_question_for_qwen(item):
    """将C-Eval格式转换为Qwen格式的问题提示"""
    question = item['question']
    options = f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
    prompt = f"{question}\n{options}\n请选择正确答案（只需回答选项字母，如A、B、C或D）："
    return prompt


def extract_answer_from_text(text):
    """从生成文本中提取答案选项（A/B/C/D）"""
    if not text:
        return None
    
    text_upper = text.upper().strip()
    
    # 方法1: 匹配单独的字母A/B/C/D（前后有空格或标点）
    pattern1 = r'\b([ABCD])\b'
    matches = re.findall(pattern1, text_upper)
    if matches:
        return matches[0]
    
    # 方法2: 匹配"答案是A"、"选择B"、"正确答案是C"等格式
    pattern2 = r'(?:答案|选择|正确答案|选项|应该选)[是：:为]\s*([ABCD])'
    matches = re.findall(pattern2, text_upper)
    if matches:
        return matches[0]
    
    # 方法3: 检查文本开头是否有选项
    if text_upper and text_upper[0] in ['A', 'B', 'C', 'D']:
        if len(text_upper) == 1 or text_upper[1] in [' ', '.', '。', ',', '，', '\n', '\t']:
            return text_upper[0]
    
    # 方法4: 匹配括号中的选项
    pattern4 = r'[\(（]([ABCD])[\)）]'
    matches = re.findall(pattern4, text_upper)
    if matches:
        return matches[0]
    
    return None


def normalize_answer(answer):
    """标准化答案格式"""
    if not answer:
        return None
    answer = str(answer).strip().upper()
    if answer in ['A', 'B', 'C', 'D']:
        return answer
    match = re.search(r'([ABCD])', answer)
    if match:
        return match.group(1)
    return None


def evaluate_ceval(model, tokenizer, ceval_data, device, max_new_tokens=128, temperature=0.1, use_chat_template=True):
    """评估C-Eval数据集"""
    model.eval()
    results = []
    predictions = []
    ground_truths = []
    
    correct_count = 0
    total_count = 0
    no_answer_count = 0
    extraction_failed_count = 0
    
    option_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predicted': 0})
    
    for item in tqdm(ceval_data, desc="Evaluating"):
        # 格式化问题
        question_text = format_question_for_qwen(item)
        
        # 使用Qwen的chat template
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": question_text}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 备用方案：直接使用问题文本
            prompt = question_text
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码生成文本（只取新生成的部分）
        generated_text = tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 提取预测答案
        predicted_answer = extract_answer_from_text(generated_text)
        
        # 获取标准答案
        ground_truth = normalize_answer(item.get('answer', ''))
        
        # 判断是否正确
        is_correct = False
        if ground_truth:
            total_count += 1
            option_stats[ground_truth]['total'] += 1
            
            if predicted_answer:
                predictions.append(predicted_answer)
                ground_truths.append(ground_truth)
                option_stats[predicted_answer]['predicted'] += 1
                
                if predicted_answer == ground_truth:
                    is_correct = True
                    correct_count += 1
                    option_stats[ground_truth]['correct'] = option_stats[ground_truth].get('correct', 0) + 1
            else:
                extraction_failed_count += 1
                predictions.append('UNKNOWN')
                ground_truths.append(ground_truth)
        else:
            no_answer_count += 1
        
        result = {
            'id': item.get('id', len(results)),
            'question': item['question'],
            'options': {
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D']
            },
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'correct': is_correct if ground_truth else None,
            'explanation': item.get('explanation', '')
        }
        
        results.append(result)
    
    # 计算指标
    metrics = {
        'total_questions': len(ceval_data),
        'questions_with_answer': total_count,
        'questions_without_answer': no_answer_count,
        'extraction_failed': extraction_failed_count,
    }
    
    if total_count > 0:
        metrics['accuracy'] = correct_count / total_count
        metrics['correct_count'] = correct_count
        metrics['error_count'] = total_count - correct_count
        
        # 各选项准确率
        option_accuracy = {}
        for option in ['A', 'B', 'C', 'D']:
            if option_stats[option]['total'] > 0:
                correct = option_stats[option].get('correct', 0)
                option_accuracy[option] = {
                    'accuracy': correct / option_stats[option]['total'],
                    'total': option_stats[option]['total'],
                    'correct': correct
                }
        metrics['option_accuracy'] = option_accuracy
        
        # 混淆矩阵
        if len(predictions) > 0 and len(ground_truths) > 0:
            valid_indices = [i for i, p in enumerate(predictions) if p != 'UNKNOWN']
            if valid_indices:
                valid_preds = [predictions[i] for i in valid_indices]
                valid_truths = [ground_truths[i] for i in valid_indices]
                
                labels = ['A', 'B', 'C', 'D']
                cm = confusion_matrix(valid_truths, valid_preds, labels=labels)
                metrics['confusion_matrix'] = cm.tolist()
                metrics['confusion_matrix_labels'] = labels
                
                # 分类报告
                try:
                    report = classification_report(
                        valid_truths, valid_preds, 
                        labels=labels, 
                        output_dict=True,
                        zero_division=0
                    )
                    metrics['classification_report'] = report
                except Exception as e:
                    logger.warning(f"Error generating classification report: {e}")
    
    return results, metrics


def save_detailed_results(results, output_path):
    """保存详细结果"""
    logger.info(f"Saving detailed results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def save_metrics(metrics, output_path):
    """保存评测指标"""
    logger.info(f"Saving metrics to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def print_summary(metrics):
    """打印评测摘要"""
    print("\n" + "="*60)
    print("C-Eval 评测结果摘要")
    print("="*60)
    print(f"总问题数: {metrics['total_questions']}")
    print(f"有答案问题数: {metrics.get('questions_with_answer', 0)}")
    print(f"无答案问题数: {metrics.get('questions_without_answer', 0)}")
    
    if metrics.get('accuracy') is not None:
        print(f"\n准确率: {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['questions_with_answer']})")
        print(f"错误数: {metrics['error_count']}")
        print(f"答案提取失败数: {metrics.get('extraction_failed', 0)}")
        
        # 各选项准确率
        if metrics.get('option_accuracy'):
            print("\n各选项准确率:")
            for option, stats in sorted(metrics['option_accuracy'].items()):
                print(f"  选项{option}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        
        # 混淆矩阵
        if metrics.get('confusion_matrix'):
            print("\n混淆矩阵:")
            labels = metrics['confusion_matrix_labels']
            cm = np.array(metrics['confusion_matrix'])
            print("      ", "  ".join([f"预测{label}" for label in labels]))
            for i, label in enumerate(labels):
                print(f"真实{label}  ", "  ".join([f"{cm[i][j]:4d}" for j in range(len(labels))]))
    
    print("="*60 + "\n")


def save_error_analysis(results, output_path):
    """保存错误分析"""
    error_results = [r for r in results if r.get('ground_truth') and not r.get('correct', False)]
    
    logger.info(f"Saving {len(error_results)} error cases to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in error_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="C-Eval格式数据集评测（适用于Qwen2.5模型）")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径（如：Qwen/Qwen2.5-3B-Instruct）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="C-Eval格式jsonl数据集路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ceval_evaluation",
        help="评测结果输出目录"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="生成温度（越低越确定，建议0.1）"
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="不使用chat template（如果模型不支持）"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    ceval_data = load_ceval_data(args.data_path)
    if not ceval_data:
        logger.error("无法加载数据")
        return
    
    logger.info(f"加载了 {len(ceval_data)} 条数据")
    
    # 检查有多少数据有答案
    with_answer = sum(1 for item in ceval_data if item.get('answer', '').strip())
    logger.info(f"其中有答案的数据: {with_answer} 条")
    
    if with_answer == 0:
        logger.warning("警告：数据集中没有答案，将无法计算准确率")
    
    # 加载模型
    device = get_device()
    logger.info(f"Loading model from {args.model_path}")
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        device_map="auto" if device.startswith('cuda') else None,
    )
    
    if not device.startswith('cuda'):
        model = model.to(device)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 检查是否支持chat template
    use_chat_template = not args.no_chat_template and hasattr(tokenizer, 'apply_chat_template')
    if use_chat_template:
        logger.info("使用chat template进行格式化")
    else:
        logger.info("不使用chat template，直接使用问题文本")
    
    # 评估
    results, metrics = evaluate_ceval(
        model, tokenizer, ceval_data, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_chat_template=use_chat_template
    )
    
    # 保存结果
    results_path = os.path.join(args.output_dir, "detailed_results.jsonl")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    error_analysis_path = os.path.join(args.output_dir, "error_analysis.jsonl")
    
    save_detailed_results(results, results_path)
    save_metrics(metrics, metrics_path)
    if metrics.get('error_count', 0) > 0:
        save_error_analysis(results, error_analysis_path)
    
    # 打印摘要
    print_summary(metrics)
    
    logger.info("评测完成！")
    logger.info(f"详细结果: {results_path}")
    logger.info(f"评测指标: {metrics_path}")
    if metrics.get('error_count', 0) > 0:
        logger.info(f"错误分析: {error_analysis_path}")


if __name__ == "__main__":
    main()