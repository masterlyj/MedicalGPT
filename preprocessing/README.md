# Medical Data Synthesis Framework

基于 LangGraph 的医疗数据集合成多智能体框架，用于生成高质量的医疗问答数据（ShareGPT格式），支持思维链（CoT）。

## 特性

- ✅ **多智能体协作**: Generator → Reviewer → Revisor 三阶段质量保证
- ✅ **多源API混合**: 支持 Groq/Gemini/DeepSeek，免费优先+付费兜底
- ✅ **异步并发**: 基于 `asyncio` 的高性能批处理
- ✅ **智能限流**: 自动故障转移，避免触及RPD限制
- ✅ **独立环境**: 使用 uv 管理依赖，与主项目隔离
- ✅ **极简日志**: 只记录进度、错误和格式问题

## 快速开始

### 1. 环境配置

```bash
cd preprocessing

# 创建虚拟环境
uv venv

# 安装依赖
uv pip install -e .

# 配置API密钥
cp .env.example .env
# 编辑 .env 填入你的API密钥
```

### 2. 准备数据

输入数据格式（JSON数组）：

```json
[
  {
    "id": "exam_001",
    "type": "exam",
    "content": "患者男，45岁。发热、咳嗽3天。X线示右下肺片状阴影。最可能的诊断是？ A. 肺炎 B. 肺癌 C. 肺结核 D. 肺栓塞",
    "answer": "A"
  },
  {
    "id": "dialogue_001",
    "type": "dialogue",
    "content": "医生，我最近总是心慌，尤其是喝咖啡后，晚上睡不着。"
  }
]
```

### 3. 运行合成

```bash
uv run scripts/synthesize.py \
  --input ../data/raw/exam_questions.json \
  --output ../data/finetune/synthesized/exam_cot.json \
  --workers 10
```

### 4. 验证结果

```bash
uv run scripts/validate.py \
  --file ../data/finetune/synthesized/exam_cot.json \
  --sample 5
```

## 项目结构

```
preprocessing/
├── config/
│   ├── models.yaml          # API配置
│   └── prompts/             # Prompt模板（YAML）
│       ├── exam_generator.yaml
│       ├── exam_reviewer.yaml
│       ├── exam_revisor.yaml
│       ├── dialogue_generator.yaml
│       └── dialogue_reviewer.yaml
│
├── src/
│   ├── state.py             # Pydantic状态模型
│   ├── workflow.py          # LangGraph工作流
│   ├── nodes.py             # Agent节点
│   ├── api_router.py        # 多源API路由
│   └── logger.py            # 极简日志
│
└── scripts/
    ├── synthesize.py        # 主合成脚本
    └── validate.py          # 质量验证脚本
```

## 输出格式

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "你是一名专业的医学专家，擅长逻辑推理和临床诊断。"
    },
    {
      "from": "human",
      "value": "患者男，45岁。发热、咳嗽3天..."
    },
    {
      "from": "gpt",
      "value": "<think>\n1. 症状分析：急性起病，发热咳嗽...\n</think>\n【答案】A\n【解析】..."
    }
  ],
  "metadata": {
    "task_id": "exam_001",
    "tokens_used": 1200,
    "api_calls": 2,
    "revision_count": 0
  }
}
```

## 成本估算

- **免费额度**: Groq (14.4k RPD), Gemini (20 RPD)
- **付费兜底**: DeepSeek (~¥0.27/M tokens)
- **10万条数据**: 约 ¥50-100（取决于免费额度使用情况）

## 故障排查

**问题**: `No providers configured for tier: fast`  
**解决**: 检查 `.env` 是否正确配置了至少一个API密钥

**问题**: `Gemini API error 429`  
**解决**: 触及免费RPD限制，系统会自动切换DeepSeek付费接口

**问题**: 生成的数据缺少 `<think>` 标签  
**解决**: 检查 `config/prompts/` 中的模板是否正确，可能需要调整system prompt

## 下一步

- 添加基于RPD的智能限流器
- 支持断点续传
- 添加人工抽检接口
- 实现多轮辩论模式
