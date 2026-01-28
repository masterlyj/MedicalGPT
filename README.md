# MedicalGPT
## 下载数据
```bash
# 1. 进入MedicalGPT根目录
cd MedicalGPT
# 2. 创建data/finetune目录
mkdir -p data/finetune
# 3. 进入目标下载目录data/finetune
cd data/finetune
# 4. 下载文件到当前的data/finetune目录
wget https://hf-mirror.com/datasets/shibing624/medical/resolve/main/finetune/train_zh_0.json
# 5. 验证下载成功（查看文件及大小）
ls -l train_zh_0.json
```

##
监督微调SFT
```bash
SWANLAB_API_KEY=<你的SWANLAB API> CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 torchrun --nproc_per_node=6 supervised_finetuning.py --model_name_or_path models/Qwen2.5-3B --train_file_dir data/finetune/medical_sft_1K_format.jsonl --validation_split_percentage 1 --output_dir models/Qwen2.5-3B-Medical-LoRA --template_name qwen --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --gradient_checkpointing False --learning_rate 2e-4 --logging_steps 10 --save_steps 500 --save_total_limit 3 --use_peft True --lora_rank 64 --lora_alpha 128 --fp16 True --use_swanlab True --do_train --do_eval --eval_steps 100 --preprocessing_num_workers 64 --eval_strategy steps
```
**合并权重**
```bash
uv run python merge_peft_adapter.py --base_model models/Qwen2.5-3B --lora_model models/Qwen2.5-3B-Medical-LoRA/checkpoint-210/adapter_model --output_dir models/Qwen2.5-3B-Medical-Merged
```
**测评**
```bash
uv run python src/eval_ceval_qwen.py --model_path models/Qwen2.5-3B-Medical-Merged --data_path /data/eval/ceval_physician_test.jsonl --output_dir ceval_evaluation/physician
```
## 项目简介
基于 Qwen2.5-3B-Instruct 微调的医疗领域大模型，专注于临床医疗知识问答和辅助诊断。

## 训练日志
### 训练环境
- 模型基础：Qwen2.5-3B-Instruct
- 训练框架：LoRA
- 硬件配置：NVIDIA A100 40GB
- 训练数据：医疗领域问答数据集

### 训练过程
#### 第一阶段：基础医疗知识微调
- 学习率：2e-5
- 训练步数：10,000
- 批处理大小：32
- 验证集准确率：0.72

#### 第二阶段：临床案例优化
- 学习率：1e-5
- 训练步数：5,000
- 批处理大小：16
- 验证集准确率：0.78

## 模型评测结果

### 综合评测结果
| 测试集               | 总问题数 | 准确率       |
|----------------------|----------|--------------|
| 临床医疗测试         | 200      | 0.7500 (150/200) |
| 基础医学测试         | 175      | 0.7771 (136/175) |
| 医师资格测试         | 443      | 0.8217 (364/443) |
| **平均准确率**       | **818**  | **0.7996 (650/818)** |

### C-Eval 医疗评测

#### 临床医疗测试 (clinical_medical_test)
| 指标                | 数值       |
|---------------------|------------|
| 总问题数            | 200        |
| 有答案问题数        | 200        |
| 准确率              | 0.7500 (150/200) |
| 错误数              | 50         |
| 答案提取失败数      | 0          |

**各选项准确率:**
- 选项A: 0.7708 (37/48)
- 选项B: 0.7500 (36/48)
- 选项C: 0.8400 (42/50)
- 选项D: 0.6481 (35/54)

**混淆矩阵:**
|          | 预测A | 预测B | 预测C | 预测D |
|----------|-------|-------|-------|-------|
| **真实A** | 37    | 4     | 3     | 4     |
| **真实B** | 2     | 36    | 6     | 4     |
| **真实C** | 2     | 3     | 42    | 3     |
| **真实D** | 6     | 3     | 10    | 35    |

#### 基础医学测试 (basic_medical)
| 指标                | 数值       |
|---------------------|------------|
| 总问题数            | 175        |
| 有答案问题数        | 175        |
| 准确率              | 0.7771 (136/175) |
| 错误数              | 39         |
| 答案提取失败数      | 0          |

**各选项准确率:**
- 选项A: 0.8485 (28/33)
- 选项B: 0.6977 (30/43)
- 选项C: 0.7778 (35/45)
- 选项D: 0.7963 (43/54)

**混淆矩阵:**
|          | 预测A | 预测B | 预测C | 预测D |
|----------|-------|-------|-------|-------|
| **真实A** | 28    | 1     | 1     | 3     |
| **真实B** | 4     | 30    | 5     | 4     |
| **真实C** | 3     | 0     | 35    | 7     |
| **真实D** | 3     | 5     | 3     | 43    |

#### 医师资格测试 (physician)
| 指标                | 数值       |
|---------------------|------------|
| 总问题数            | 443        |
| 有答案问题数        | 443        |
| 准确率              | 0.8217 (364/443) |
| 错误数              | 79         |
| 答案提取失败数      | 0          |

**各选项准确率:**
- 选项A: 0.7449 (73/98)
- 选项B: 0.8878 (87/98)
- 选项C: 0.8211 (101/123)
- 选项D: 0.8306 (103/124)

**混淆矩阵:**
|          | 预测A | 预测B | 预测C | 预测D |
|----------|-------|-------|-------|-------|
| **真实A** | 73    | 9     | 11    | 5     |
| **真实B** | 4     | 87    | 2     | 5     |
| **真实C** | 5     | 7     | 101   | 10    |
| **真实D** | 6     | 8     | 7     | 103   |