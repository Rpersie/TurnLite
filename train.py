import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import argparse
from pathlib import Path
from data_processor import DataProcessor, prepare_training_data
# 1. 导入TensorBoard相关依赖
from torch.utils.tensorboard import SummaryWriter
import datetime


class QwenFineTuner:
    def __init__(self, config):
        self.config = config
        self.load_system_prompt()
        self.setup_environment()  # 改为初始化TensorBoard
        self.load_model_and_tokenizer()
        # 2. 初始化TensorBoard写入器（绑定日志目录）
        self.init_tensorboard_writer()
    
    def load_system_prompt(self):
        """Load system prompt from file（逻辑不变）"""
        if "system_prompt_file" in self.config:
            prompt_file = self.config["system_prompt_file"]
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self.config["system_prompt"] = f.read().strip()
                print(f"Loaded system prompt from: {prompt_file}")
            else:
                raise FileNotFoundError(f"System prompt file not found: {prompt_file}")
        elif "system_prompt" not in self.config:
            raise ValueError("Either 'system_prompt' or 'system_prompt_file' must be specified in config")
    
    def setup_environment(self):
        """3. 移除SwanLab环境配置，保留GPU设置"""
        # Set GPU environment if specified
        if "gpu_ids" in self.config and self.config["gpu_ids"]:
            gpu_ids = ",".join(map(str, self.config["gpu_ids"]))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print(f"Using GPUs: {gpu_ids}")
        
        # 打印关键配置（替代原SwanLab.config的记录功能）
        print("Experiment Config:")
        config_keys = ["model", "prompt", "data_max_length", "dataset_version", "device", "torch_dtype"]
        for key in config_keys:
            if key in self.config:
                print(f"  {key}: {self.config[key]}")
    
    def init_tensorboard_writer(self):
        """4. 初始化TensorBoard Writer，按实验名创建唯一日志目录"""
        # 生成日志目录：output_dir/tensorboard/experiment_name_时间戳
        experiment_name = self.config.get("experiment_name", "default")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(
            self.config["output_dir"], 
            "tensorboard", 
            f"{experiment_name}_{timestamp}"
        )
        os.makedirs(tb_log_dir, exist_ok=True)  # 确保目录存在
        
        # 创建Writer实例
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard log directory created: {tb_log_dir}")
        
        # 5. 记录超参数到TensorBoard（替代swanlab.config）
        hparams = {
            "model_name": self.config["model_name"],
            "model_path": self.config["model_path"],
            "max_length": self.config["max_length"],
            "dataset_version": self.config["dataset_version"],
            "train_batch_size": self.config.get("train_batch_size", 20),
            "eval_batch_size": self.config.get("eval_batch_size", 20),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 4),
            "num_epochs": self.config.get("num_epochs", 2),
            "learning_rate": self.config.get("learning_rate", 1e-4),
            "eval_steps": self.config.get("eval_steps", 30),
            "logging_steps": self.config.get("logging_steps", 10),
            "seed": self.config.get("seed", 42),
            "torch_dtype": self.config.get("torch_dtype", "float16"),
            "eval_split_ratio": self.config.get("eval_split_ratio", 0.1),
        }
        # 超参数写入TensorBoard（需搭配一个初始指标，这里用学习率）
        self.tb_writer.add_hparams(hparams, {"hparams/learning_rate": hparams["learning_rate"]})
    
    def load_model_and_tokenizer(self):
        """逻辑不变，仅移除SwanLab相关打印"""
        print(f"Loading model from: {self.config['model_path']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"], 
            use_fast=False, 
            trust_remote_code=True
        )
        
        # Determine torch dtype
        torch_dtype_str = self.config.get("torch_dtype", "float16")
        if torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16
            print(f"Unknown torch_dtype '{torch_dtype_str}', using float16")
        
        # Determine device map
        device_map = self.config.get("device", "auto")
        
        print(f"Loading model with device_map='{device_map}', torch_dtype={torch_dtype}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_path"], 
            device_map=device_map, 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        self.model.enable_input_require_grads()
    
    def prepare_datasets(self):
        """逻辑完全不变"""
        data_fields = self.config.get("data_fields", {"input": "question", "output": "answer"})
        eval_split_ratio = self.config.get("eval_split_ratio", 0.1)
        
        full_dataset = prepare_training_data(
            raw_path=self.config["train_raw_path"],
            formatted_path=self.config["train_formatted_path"],
            system_prompt_file=self.config["system_prompt_file"],
            tokenizer=self.tokenizer,
            max_length=self.config["max_length"],
            data_fields=data_fields,
            shuffle_seed=self.config.get("seed", 42)
        )
        
        total_size = len(full_dataset)
        val_size = int(total_size * eval_split_ratio)
        train_size = total_size - val_size
        
        self.train_dataset = full_dataset.select(range(train_size))
        self.val_dataset = full_dataset.select(range(train_size, total_size))
        
        print(f"Total samples: {total_size}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)} (split ratio: {eval_split_ratio:.2f})")
    
    def setup_training_args(self):
        """6. 修改TrainingArguments：移除SwanLab报告，添加TensorBoard配置"""
        experiment_name = self.config.get("experiment_name", "default")
        model_name = self.config.get("model_name", "Qwen2.5-1.5B-Instruct").split("/")[-1]
        
        # Generate run_name based on experiment_name
        if "run_name" not in self.config or self.config["run_name"] == "Qwen2.5-1.5B-Instruct":
            run_name = f"{model_name}-{experiment_name}"
        else:
            run_name = self.config["run_name"]
        
        return TrainingArguments(
            output_dir=self.config["output_dir"],
            per_device_train_batch_size=self.config.get("train_batch_size", 20),
            per_device_eval_batch_size=self.config.get("eval_batch_size", 20),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            eval_strategy="steps",
            eval_steps=self.config.get("eval_steps", 30),
            logging_steps=self.config.get("logging_steps", 10),
            num_train_epochs=self.config.get("num_epochs", 2),
            save_steps=self.config.get("save_steps", 40),
            learning_rate=self.config.get("learning_rate", 1e-4),
            save_on_each_node=True,
            gradient_checkpointing=True,
            # 关键修改：将report_to从"swanlab"改为"tensorboard"
            report_to="tensorboard",
            # 绑定TensorBoard日志目录（与init_tensorboard_writer保持一致）
            logging_dir=os.path.join(self.config["output_dir"], "tensorboard"),
            run_name=run_name,
        )
    
    def train(self):
        """7. 训练逻辑不变，仅确保Trainer使用TensorBoard"""
        self.prepare_datasets()
        training_args = self.setup_training_args()
        
        # Trainer会自动通过TrainingArguments的report_to和logging_dir连接TensorBoard
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )
        
        print("Starting training...")
        trainer.train()
        print("Training completed!")
        
        # 8. 训练结束后关闭TensorBoard Writer（避免资源泄漏）
        self.tb_writer.close()
        print("TensorBoard Writer closed successfully")
    
    def predict(self, messages):
        """逻辑完全不变（预测核心逻辑）"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.config.get("max_new_tokens", 512),
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def evaluate_samples(self, num_samples=3):
        """9. 替换SwanLab文本记录为TensorBoard文本日志"""
        test_df = pd.read_json(self.config["val_formatted_path"], lines=True)[:num_samples]
        
        for index, row in test_df.iterrows():
            messages = [
                {"role": "system", "content": row['instruction']},
                {"role": "user", "content": row['input']}
            ]
            
            response = self.predict(messages)
            
            # 构造评估文本（保留原格式）
            response_text = f"""
Sample {index + 1}:
Question: {row['input']}
Expected: {row['output']}
LLM Response: {response}
==================================================
            """
            # 10. 记录文本到TensorBoard（替代swanlab.log(Text)）
            # 用step=index区分不同样本，tag统一为"eval/sample_predictions"
            self.tb_writer.add_text(
                tag="eval/sample_predictions",
                text_string=response_text,
                global_step=index
            )
            print(response_text)


def get_default_config():
    """11. 移除默认配置中的SwanLab相关字段"""
    return {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_path": "./models/Qwen2.5-1.5B-Instruct/Qwen/Qwen2.5-1.5B-Instruct",
        "system_prompt_file": "system_prompt.txt",
        "max_length": 2048,
        "dataset_version": "Qifu-Datset-Version-6",
        "experiment_name": "default",  # 保留实验名用于TensorBoard目录
        
        # Training parameters
        "train_batch_size": 20,
        "eval_batch_size": 20,
        "gradient_accumulation_steps": 4,
        "num_epochs": 2,
        "learning_rate": 1e-4,
        "eval_steps": 30,
        "logging_steps": 10,
        "save_steps": 40,
        "seed": 42,
        "run_name": "Qwen2.5-1.5B-Instruct",
        "eval_split_ratio": 0.1,
    }


def parse_args():
    """12. 移除命令行参数中的SwanLab相关字段（如swanlab_project）"""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--model_path", type=str, help="Path to model")
    parser.add_argument("--train_raw_path", type=str, help="Path to raw training data")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--dataset_version", type=str, help="Dataset version")
    parser.add_argument("--experiment_name", type=str, help="Experiment name for organizing outputs")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, help="Evaluation batch size")
    parser.add_argument("--eval_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, help="Device mapping (auto, cuda, cpu)")
    parser.add_argument("--torch_dtype", type=str, help="Torch dtype (float16, bfloat16, float32)")
    parser.add_argument("--gpu_ids", type=str, help="GPU IDs to use (comma-separated, e.g., '0,1')")
    parser.add_argument("--eval_split_ratio", type=float, help="Ratio of training data to use for validation (default: 0.1)")
    
    return parser.parse_args()


def expand_config_variables(config):
    """逻辑不变（变量替换功能保留）"""
    experiment_name = config.get("experiment_name", "default")
    dataset_version = config.get("dataset_version", "unknown")
    
    variables = {
        "experiment_name": experiment_name,
        "dataset_version": dataset_version,
    }
    
    def expand_value(value):
        if isinstance(value, str):
            try:
                return value.format(**variables)
            except KeyError as e:
                print(f"Warning: Unknown variable {e} in config value: {value}")
                return value
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(item) for item in value]
        else:
            return value
    
    return expand_value(config)


def load_config(config_path):
    """逻辑不变（配置加载功能保留）"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        print(f"Config file {config_path} not found, using default config")
        config = get_default_config()
    
    config = expand_config_variables(config)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            if key == 'gpu_ids' and isinstance(value, str):
                config[key] = [int(x.strip()) for x in value.split(',')]
            else:
                config[key] = value
    
    # Validate required paths
    required_paths = ["train_raw_path", "output_dir", "train_formatted_path", "val_formatted_path"]
    for path_key in required_paths:
        if path_key not in config:
            raise ValueError(f"{path_key} must be specified in config")
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize fine-tuner
    fine_tuner = QwenFineTuner(config)
    
    # Start training
    fine_tuner.train()


if __name__ == "__main__":
    main()