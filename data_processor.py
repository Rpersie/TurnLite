import json
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset


class DataProcessor:
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt
    
    def convert_raw_to_format(self, raw_path, formatted_path, data_fields=None):
        """
        Convert raw JSONL data to training/evaluation format
        
        Args:
            raw_path: Path to raw JSONL file
            formatted_path: Path to save formatted JSONL file
            data_fields: Dict mapping field names, e.g., {"input": "question", "output": "answer"}
        """
        if data_fields is None:
            data_fields = {"input": "question", "output": "answer"}
        
        if os.path.exists(formatted_path):
            print(f"Formatted data already exists: {formatted_path}")
            return formatted_path
        
        messages = []
        
        print(f"Converting {raw_path} to {formatted_path}")
        
        with open(raw_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract input and output based on field mapping
                    input_text = data.get(data_fields["input"], "")
                    output_text = data.get(data_fields["output"], "")
                    
                    if not input_text or not output_text:
                        print(f"Warning: Missing data in line {line_num}, skipping")
                        continue
                    
                    message = {
                        "instruction": self.system_prompt,
                        "input": input_text,
                        "output": output_text,
                    }
                    messages.append(message)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
        
        if not messages:
            raise ValueError(f"No valid data found in {raw_path}")
        
        # Create directory if it doesn't exist
        Path(formatted_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save formatted data
        with open(formatted_path, "w", encoding="utf-8") as file:
            for message in messages:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")
        
        print(f"Converted {len(messages)} samples to {formatted_path}")
        return formatted_path
    
    def load_formatted_data(self, formatted_path, sample_size=-1):
        """
        Load formatted JSONL data
        
        Args:
            formatted_path: Path to formatted JSONL file
            sample_size: Number of samples to load (-1 for all)
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if not os.path.exists(formatted_path):
            raise FileNotFoundError(f"Formatted data file not found: {formatted_path}")
        
        df = pd.read_json(formatted_path, lines=True)
        
        if sample_size > 0:
            df = df.head(sample_size)
        
        print(f"Loaded {len(df)} samples from {formatted_path}")
        return df
    
    def prepare_dataset_for_training(self, formatted_path, tokenizer, max_length=2048, shuffle_seed=42):
        """
        Prepare dataset for training (with tokenization)
        
        Args:
            formatted_path: Path to formatted JSONL file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            shuffle_seed: Random seed for shuffling
        
        Returns:
            datasets.Dataset: Processed dataset ready for training
        """
        df = self.load_formatted_data(formatted_path)
        ds = Dataset.from_pandas(df)
        
        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed)
        
        # Process function for tokenization
        def process_func(example):
            instruction = tokenizer(
                f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )
            response = tokenizer(f"{example['output']}", add_special_tokens=False)
            
            input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
            
            # Truncate if too long
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
        processed_ds = ds.map(process_func, remove_columns=ds.column_names)
        print(f"Processed {len(processed_ds)} samples for training")
        
        return processed_ds
    
    def prepare_data_for_evaluation(self, formatted_path, sample_size=-1):
        """
        Prepare data for evaluation (no tokenization, just load and format)
        
        Args:
            formatted_path: Path to formatted JSONL file
            sample_size: Number of samples to load (-1 for all)
        
        Returns:
            pandas.DataFrame: Data ready for evaluation
        """
        return self.load_formatted_data(formatted_path, sample_size)
    
    def auto_convert_and_load(self, raw_path, formatted_path, data_fields=None, for_training=False, 
                             tokenizer=None, max_length=2048, sample_size=-1, shuffle_seed=42):
        """
        Auto convert raw data to formatted data and load it
        
        Args:
            raw_path: Path to raw JSONL file
            formatted_path: Path to save/load formatted JSONL file
            data_fields: Field mapping for conversion
            for_training: Whether to prepare for training (requires tokenizer)
            tokenizer: Tokenizer instance (required if for_training=True)
            max_length: Maximum sequence length (for training)
            sample_size: Number of samples to load (-1 for all)
            shuffle_seed: Random seed for shuffling (for training)
        
        Returns:
            Dataset or DataFrame: Processed data
        """
        # Convert if needed
        self.convert_raw_to_format(raw_path, formatted_path, data_fields)
        
        # Load based on purpose
        if for_training:
            if tokenizer is None:
                raise ValueError("Tokenizer is required for training data preparation")
            return self.prepare_dataset_for_training(formatted_path, tokenizer, max_length, shuffle_seed)
        else:
            return self.prepare_data_for_evaluation(formatted_path, sample_size)


def load_system_prompt(prompt_file_path):
    """Load system prompt from file"""
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        print(f"Warning: System prompt file not found: {prompt_file_path}")
        return ""


# Convenience functions for common use cases
def prepare_training_data(raw_path, formatted_path, system_prompt_file, tokenizer, 
                         max_length=2048, data_fields=None, shuffle_seed=42):
    """Convenience function for preparing training data"""
    system_prompt = load_system_prompt(system_prompt_file)
    processor = DataProcessor(system_prompt)
    return processor.auto_convert_and_load(
        raw_path=raw_path,
        formatted_path=formatted_path,
        data_fields=data_fields,
        for_training=True,
        tokenizer=tokenizer,
        max_length=max_length,
        shuffle_seed=shuffle_seed
    )


def prepare_evaluation_data(raw_path, formatted_path, system_prompt_file, 
                           sample_size=-1, data_fields=None):
    """Convenience function for preparing evaluation data"""
    system_prompt = load_system_prompt(system_prompt_file)
    processor = DataProcessor(system_prompt)
    return processor.auto_convert_and_load(
        raw_path=raw_path,
        formatted_path=formatted_path,
        data_fields=data_fields,
        for_training=False,
        sample_size=sample_size
    )