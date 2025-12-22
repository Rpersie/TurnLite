import json
import pandas as pd
import torch
import os
import argparse
import numpy as np
from datasets import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from data_processor import DataProcessor, prepare_evaluation_data


class QwenEvaluator:
    def __init__(self, config):
        self.config = config
        self.load_system_prompt()
        self.setup_environment()
        self.load_model_and_tokenizer()
        
        # 定义类别顺序
        self.classes = ["finished", "unfinished", "wait"]
    
    def load_system_prompt(self):
        """Load system prompt from file"""
        if "system_prompt_file" in self.config:
            prompt_file = self.config["system_prompt_file"]
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self.config["system_prompt"] = f.read().strip()
                print(f"Loaded system prompt from: {prompt_file}")
            else:
                print(f"Warning: System prompt file not found: {prompt_file}, using empty prompt")
                self.config["system_prompt"] = ""
        elif "system_prompt" not in self.config:
            self.config["system_prompt"] = ""
    
    def setup_environment(self):
        """Setup environment variables"""
        # Set GPU environment if specified
        if "gpu_ids" in self.config and self.config["gpu_ids"]:
            gpu_ids = ",".join(map(str, self.config["gpu_ids"]))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print(f"Using GPUs: {gpu_ids}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        print(f"Loading model from: {self.config['model_path']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"], 
            use_fast=False, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine torch dtype
        torch_dtype_str = self.config.get("torch_dtype", "auto")
        if torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"
        
        # Determine device map
        device_map = "cpu" if self.config.get("cpu_only", False) else self.config.get("device", "auto")
        
        print(f"Loading model with device_map='{device_map}', torch_dtype={torch_dtype}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_path"], 
            device_map=device_map, 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()
        
        self.model.generation_config.max_new_tokens = self.config.get("max_new_tokens", 512)
    
    def load_test_data(self):
        """Load test data - supports both raw and formatted data with caching"""
        # Check if we have raw data path and need to convert
        if "test_raw_path" in self.config:
            # Auto-generate test_formatted_path based on output directory
            output_dir = Path(self.config["output_dir"])
            test_formatted_path = output_dir / "test_format.jsonl"
            
            # Use shared data processor to handle conversion (with caching)
            data_fields = self.config.get("data_fields", {"input": "question", "output": "answer"})
            test_df = prepare_evaluation_data(
                raw_path=self.config["test_raw_path"],
                formatted_path=str(test_formatted_path),
                system_prompt_file=self.config["system_prompt_file"],
                sample_size=self.config.get("sample_size", -1),
                data_fields=data_fields
            )
        else:
            # Fallback to direct loading (for backward compatibility)
            test_path = self.config.get("test_data_path", self.config.get("test_formatted_path"))
            if not test_path or not os.path.exists(test_path):
                raise FileNotFoundError(f"Test data file not found: {test_path}")
            
            test_df = pd.read_json(test_path, lines=True)
            
            # Apply sample limit if specified
            sample_size = self.config.get("sample_size", -1)
            if sample_size > 0:
                test_df = test_df.head(sample_size)
            
            print(f"Loaded {len(test_df)} test samples from {test_path}")
        
        return test_df
    
    def preprocess_and_cache_data(self):
        """Preprocess all data and cache the formatted prompts"""
        test_df = self.load_test_data()
        
        # Check if cached preprocessed data exists
        cache_path = Path(self.config["output_dir"]) / "preprocessed_prompts.json"
        
        if cache_path.exists():
            print(f"Loading cached preprocessed data from: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Verify cache is still valid (same number of samples)
            if len(cached_data) == len(test_df):
                return cached_data, test_df
            else:
                print("Cache size mismatch, regenerating...")
        
        print("Preprocessing and caching prompts...")
        preprocessed_data = []
        
        for index, row in test_df.iterrows():
            if index % 100 == 0:
                print(f"Preprocessing sample {index + 1}/{len(test_df)}")
            
            # Prepare messages
            if self.config["system_prompt"]:
                messages = [
                    {"role": "system", "content": self.config["system_prompt"]},
                    {"role": "user", "content": row['input']}
                ]
            else:
                messages = [
                    {"role": "user", "content": row['input']}
                ]
            
            # Convert to text format (this is the expensive operation)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            preprocessed_data.append({
                "index": index,
                "formatted_text": text,
                "true_label": row['output'].lower(),
                "original_input": row['input']
            })
        
        # Cache the preprocessed data
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Cached preprocessed data to: {cache_path}")
        return preprocessed_data, test_df
    
    def predict(self, messages):
        """Generate prediction for given messages"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # 只使用max_new_tokens，让模型自动计算max_length
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 32),
                do_sample=False,
                # temperature=0.0,  # 显式设置温度参数为固定值0.0
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def predict_from_text(self, formatted_text):
        """Generate prediction from pre-formatted text (faster)"""
        inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # 最简化配置，使用库的默认优化路径
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 32),
                # pad_token_id=self.tokenizer.eos_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
                # use_cache=True,
                # num_beams=1,
                # early_stopping=True
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def run_evaluation(self):
        """Run evaluation on test data with caching"""
        # Use cached preprocessing
        preprocessed_data, test_df = self.preprocess_and_cache_data()
        results = []
        
        print("Starting evaluation with cached data...")
        for i, item in enumerate(preprocessed_data):
            if i % 50 == 0:
                print(f"Processing sample {i + 1}/{len(preprocessed_data)}")
            
            # Get prediction using pre-formatted text (much faster!)
            pred_label = self.predict_from_text(item["formatted_text"]).lower()
            true_label = item["true_label"]
            
            # Save result
            results.append({
                "question": item["original_input"],
                "true_label": true_label,
                "pred_label": pred_label
            })
        
        print(f"Evaluation completed! Processed {len(results)} samples")
        return results
    
    def save_results(self, results):
        """Save evaluation results"""
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_path = output_dir / "evaluation_results.jsonl"
        with open(results_path, 'w', encoding='utf-8') as f:
            for item in results:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"Detailed results saved to: {results_path}")
        
        return results_path
    
    def analyze_results(self, results):
        """Analyze results and generate metrics"""
        output_dir = Path(self.config["output_dir"])
        
        # Extract labels
        y_true = [item["true_label"] for item in results]
        y_pred = [item["pred_label"] for item in results]
        
        # Filter valid labels
        valid_indices = [i for i in range(len(y_true)) 
                        if y_true[i] in self.classes and y_pred[i] in self.classes]
        
        if not valid_indices:
            print("Warning: No valid labels found for analysis")
            return
        
        y_true_filtered = [y_true[i] for i in valid_indices]
        y_pred_filtered = [y_pred[i] for i in valid_indices]
        
        print(f"Analyzing {len(valid_indices)} valid samples out of {len(results)} total")
        
        # Calculate metrics
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=self.classes)
        report = classification_report(
            y_true_filtered, y_pred_filtered,
            labels=self.classes,
            target_names=self.classes,
            output_dict=False,
            digits=2
        )
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {len(results)}")
        print(f"Valid samples: {len(valid_indices)}")
        print(f"Classes: {self.classes}")
        
        print("\nConfusion Matrix:")
        print("Rows: True labels, Columns: Predicted labels")
        print(cm)
        
        print("\nClassification Report:")
        print(report)
        
        # Save report
        report_path = output_dir / "classification_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Total samples: {len(results)}\n")
            f.write(f"Valid samples: {len(valid_indices)}\n")
            f.write(f"Classes: {self.classes}\n\n")
            f.write("Confusion Matrix:\n")
            f.write("Rows: True labels, Columns: Predicted labels\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
        
        return cm, report
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to: {save_path}")


def get_default_config():
    """Get default evaluation configuration"""
    return {
        "model_path": "./exp/finetune-kimi-v3-data/model/checkpoint-320",
        "system_prompt_file": "./data/prompt/system_prompt.txt",
        "test_data_path": "./exp/finetune-kimi-v3-data/data/val_format.jsonl",
        "output_dir": "./evaluation_results",
        
        "device": "auto",
        "torch_dtype": "float16",
        "gpu_ids": [0],
        "cpu_only": False,
        
        "max_new_tokens": 32,
        "sample_size": -1,  # -1 means use all samples
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Qwen model")
    parser.add_argument("--config", type=str, default="eval_config.json", help="Path to config file")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--test_data_path", type=str, help="Path to test data")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--device", type=str, help="Device mapping (auto, cuda, cpu)")
    parser.add_argument("--torch_dtype", type=str, help="Torch dtype (float16, bfloat16, float32)")
    parser.add_argument("--gpu_ids", type=str, help="GPU IDs to use (comma-separated)")
    parser.add_argument("--cpu_only", action="store_true", help="Use CPU only")
    
    return parser.parse_args()


def extract_testset_name(config):
    """Extract clean testset name from test data path"""
    testset_name = "unknown_testset"
    
    # Try to extract from test_raw_path first
    if "test_raw_path" in config:
        test_path = config["test_raw_path"]
        filename = os.path.basename(test_path)
        testset_name = os.path.splitext(filename)[0]
    
    # Try to extract from test_data_path as fallback
    elif "test_data_path" in config:
        test_path = config["test_data_path"]
        filename = os.path.basename(test_path)
        testset_name = os.path.splitext(filename)[0]
    
    # Try to extract from test_formatted_path as last resort
    elif "test_formatted_path" in config:
        test_path = config["test_formatted_path"]
        filename = os.path.basename(test_path)
        testset_name = os.path.splitext(filename)[0]
        # Remove common suffixes like "_format"
        testset_name = testset_name.replace("_format", "").replace("-format", "")
    
    # Clean up the name
    testset_name = testset_name.replace('/', '-').replace('\\', '-')
    return testset_name


def extract_checkpoint_name(config):
    """Extract checkpoint name from model path"""
    checkpoint_name = "unknown_checkpoint"
    
    if "model_path" in config:
        model_path = config["model_path"]
        path_parts = model_path.split('/')
        
        # Look for checkpoint info
        for part in path_parts:
            if 'checkpoint' in part.lower():
                checkpoint_name = part
                break
        else:
            # Look for model version info
            for part in path_parts:
                if any(keyword in part.lower() for keyword in ['final', 'best', 'model']):
                    checkpoint_name = part
                    break
            else:
                # Fallback: use the last non-empty directory name
                checkpoint_name = [p for p in path_parts if p][-1] if path_parts else "unknown_checkpoint"
    
    # Clean up the name
    checkpoint_name = checkpoint_name.replace('/', '-').replace('\\', '-')
    return checkpoint_name


def generate_output_dir(config):
    """Generate output directory: experiment_name/testset_name/checkpoint_name"""
    # Get experiment name
    experiment_name = config.get("experiment_name", "default_experiment")
    
    # Extract testset and checkpoint names
    testset_name = extract_testset_name(config)
    checkpoint_name = extract_checkpoint_name(config)
    
    # Generate structured output directory
    output_dir = f"./evaluation_results/{experiment_name}/{testset_name}/{checkpoint_name}"
    return output_dir


def expand_config_variables(config):
    """Expand variables in config values (e.g., {experiment_name})"""
    experiment_name = config.get("experiment_name", "default")
    dataset_version = config.get("dataset_version", "unknown")
    
    # Variables that can be used in config
    variables = {
        "experiment_name": experiment_name,
        "dataset_version": dataset_version,
    }
    
    # Recursively expand variables in config values
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
    """Load configuration from JSON file and expand variables"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        print(f"Config file {config_path} not found, using default config")
        config = get_default_config()
    
    # Expand variables in config
    config = expand_config_variables(config)
    
    # Auto-generate output directory if not specified or using default
    if config.get("output_dir") == "./evaluation_results" or "output_dir" not in config:
        config["output_dir"] = generate_output_dir(config)
        print(f"Auto-generated output directory: {config['output_dir']}")
    
    return config


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            if key == 'gpu_ids' and isinstance(value, str):
                config[key] = [int(x.strip()) for x in value.split(',')]
            else:
                config[key] = value
    
    print("Evaluation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize evaluator
    evaluator = QwenEvaluator(config)
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Save results
    evaluator.save_results(results)
    
    # Analyze results
    evaluator.analyze_results(results)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()