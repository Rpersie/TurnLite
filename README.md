# Turn-lite: Conversational Turn-taking Detection

Turn-lite 是一个专为会话轮次检测（Turn-taking Detection）设计的轻量级框架。它基于 Qwen 系列大语言模型，支持从原始数据处理、模型微调到自动化评估的全流程。

## 核心功能

*   **模型支持**：深度集成 Hugging Face `transformers`，针对 Qwen2.5 系列模型进行了优化。
*   **多阶段训练支持**：项目支持从预训练 (Pretrain)、指令微调 (Finetune) 到思维链微调 (CoT) 的多阶段训练模式。
*   **状态分类**：支持将说话人的状态分为三类：
    *   `finished`: 说话已完成
    *   `unfinished`: 说话未完成（处于句中或需继续）
    *   `wait`: 等待状态或停顿
*   **思维链支持 (CoT)**：支持推理型 Prompt，允许模型在给出结论前进行思考，提升复杂场景下的检测效果。
*   **自动化流水线**：
    *   `data_processor.py`: 自动处理原始 JSONL 格式，支持自定义字段映射。
    *   `train.py`: 集成 `Trainer` 的高效微调脚本，支持 TensorBoard 日志记录。
    *   `evaluate_model.py`: 自动化的评估工具，生成详细的分类报告和混淆矩阵。

## 快速开始

### 环境依赖

*   Python 3.8+
*   PyTorch 2.0+
*   Transformers, Datasets, PEFT
*   TensorBoard
*   Scikit-learn, Pandas, Matplotlib, Seaborn (评估相关)

### 目录结构

```text
Turn-lite/
├── configs/           # 训练与评估的配置文件 (JSON)
├── data/              # 存放原始数据及处理后数据
├── data_processor.py  # 数据转换与 Tokenizer 处理
├── train.py           # 模型微调主脚本
├── evaluate_model.py  # 模型评估与指标生成脚本
└── README.md          # 项目说明
```

## 使用指南

### 1. 数据准备

原始数据推荐使用 JSONL 格式。你可以在配置文件中指定 `data_fields` 来映射字段：

```json
{"question": "今天的天气真的很", "answer": "unfinished"}
{"question": "请帮我预定下午的会议。", "answer": "finished"}
```

### 2. 模型训练

使用配置文件启动训练：

```bash
python train.py --config configs/train/your_train_config.json
```

关键配置项：
*   `model_path`: 预训练模型路径
*   `train_raw_path`: 原始训练数据路径
*   `output_dir`: 输出目录（存放 checkpoint 和 TensorBoard 日志）

### 3. 模型评估

对训练好的模型进行全面评估：

```bash
python evaluate_model.py --config configs/test/eval_config.json --model_path ./exp/checkpoint-1000
```

评估完成后，会在 `evaluation_results/` 目录下生成：
*   `evaluation_results.jsonl`: 详细的每条预测结果。
*   `classification_report.txt`: 标准分类指标（精确率、召回率、F1值）。
*   `confusion_matrix.png`: 混淆矩阵热图，直观展示分类效果。

## 配置示例

在测试配置文件中，你可以灵活控制评估行为：

```json
{
    "experiment_name": "turnlite-qwen-cot",
    "system_prompt_file": "./data/prompt/system_prompt.txt",
    "data_fields": {
        "input": "question",
        "output": "answer"
    },
    "max_new_tokens": 128,
    "torch_dtype": "bfloat16",
    "gpu_ids": [0]
}
```

## 许可证

[MIT License](LICENSE)
