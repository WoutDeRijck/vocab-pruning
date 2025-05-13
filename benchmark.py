import os
import time
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import gc
import json
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader

class TokenMappingWrapper(nn.Module):
    """Wrapper for models with token mapping to handle OOV tokens during inference."""
    
    def __init__(self, model, token_map=None, pruned_vocab_size=None):
        super().__init__()
        self.model = model
        self.token_map = token_map
        self.pruned_vocab_size = pruned_vocab_size or model.model.embeddings.tok_embeddings.num_embeddings
        self.mapping_time = 0
        self.mapping_calls = 0
    
    def forward(self, **inputs):
        if 'input_ids' in inputs and self.token_map is not None:
            # Time the token mapping process
            mapping_start = time.time()
            
            # Apply token mapping to the input_ids tensor
            input_ids = inputs['input_ids']
            mapped_input_ids = torch.zeros_like(input_ids)
            
            # Use a mask to identify padding tokens (usually 0)
            padding_mask = (input_ids == 0)
            
            # Convert to lists for easier processing
            input_ids_list = input_ids.tolist()
            
            # Map tokens using the token mapping, using 0 (UNK) for OOV tokens
            for batch_idx, sequence in enumerate(input_ids_list):
                for seq_idx, token_id in enumerate(sequence):
                    if token_id in self.token_map:
                        mapped_input_ids[batch_idx, seq_idx] = self.token_map[token_id]
                    else:
                        mapped_input_ids[batch_idx, seq_idx] = 0  # UNK token
            
            # Restore padding tokens
            mapped_input_ids = mapped_input_ids.masked_fill(padding_mask, 0)
            
            # Replace the input_ids with the mapped version
            inputs['input_ids'] = mapped_input_ids
            
            # Ensure all token IDs are within the pruned vocabulary range
            if torch.max(mapped_input_ids) >= self.pruned_vocab_size:
                print(f"Warning: Mapped input_ids contain tokens >= vocab size {self.pruned_vocab_size}")
                print(f"Max token ID: {torch.max(mapped_input_ids).item()}")
                # Clip to max vocab size - 1
                inputs['input_ids'] = torch.clamp(mapped_input_ids, 0, self.pruned_vocab_size - 1)
            
            # Record the mapping time
            mapping_end = time.time()
            self.mapping_time += (mapping_end - mapping_start)
            self.mapping_calls += 1
        
        # Forward the mapped inputs to the actual model
        return self.model(**inputs)
    
    def get_avg_mapping_time(self):
        """Return the average time spent on token mapping."""
        if self.mapping_calls == 0:
            return 0
        return self.mapping_time / self.mapping_calls

def get_model_size(model_path):
    """Get the size of a model directory or file in MB, counting only model weights"""
    model_files = [
        "pytorch_model.bin",
        "model.bin", 
        "model.safetensors",
        "pytorch_model.safetensors"
    ]
    
    # Files to explicitly exclude from size calculation
    exclude_files = [
        "optimizer.pt",
        "optimizer.pth",
        "optimizer.bin",
        "scheduler.pt",
        "scheduler.bin",
        "training_args.bin",
        "trainer_state.json",
        "rng_state.pth"
    ]
    
    if os.path.isdir(model_path):
        total_size = 0
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                # If it's an exact model file or has a model file extension but not in exclude list
                if (f in model_files or 
                    (f.endswith(".bin") or f.endswith(".pt") or f.endswith(".safetensors")) and 
                    not any(f.endswith(exclude) for exclude in exclude_files)):
                    fp = os.path.join(dirpath, f)
                    size = os.path.getsize(fp)
                    print(f"Including file in size calculation: {fp} ({size / (1024 * 1024):.2f} MB)")
                    total_size += size
                elif f in exclude_files or any(f.endswith(exclude) for exclude in exclude_files):
                    fp = os.path.join(dirpath, f)
                    size = os.path.getsize(fp)
                    print(f"Excluding file from size calculation: {fp} ({size / (1024 * 1024):.2f} MB)")
        return total_size / (1024 * 1024)
    else:
        return os.path.getsize(model_path) / (1024 * 1024)

def measure_gpu_memory(model, tokenizer, dataset, batch_size=16, max_samples=100, token_map=None):
    """Measure peak GPU memory usage during inference"""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    
    # Record starting GPU memory
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.memory_allocated()
    
    # Wrap model with token mapping if provided
    if token_map is not None:
        wrapped_model = TokenMappingWrapper(model, token_map)
        model_to_use = wrapped_model
    else:
        model_to_use = model
    
    # Move model to GPU
    model_to_use = model_to_use.cuda()
    
    # Create DataLoader with small subset of dataset
    limited_dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def collate_fn(batch):
        # Extract text fields based on dataset format
        if 'sentence' in batch[0]:  # CoLA
            texts = [item['sentence'] for item in batch]
        elif 'sentence1' in batch[0]:  # MRPC, MNLI, etc.
            texts = [f"{item['sentence1']} {item['sentence2']}" for item in batch]
        else:
            texts = [str(item) for item in batch]  # Fallback
        
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    dataloader = DataLoader(limited_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Run inference on batches
    with torch.no_grad():
        for inputs in dataloader:
            # Move inputs to GPU
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model_to_use(**inputs)
    
    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = peak_memory - starting_memory
    
    # Move model back to CPU to clear GPU memory
    model_to_use = model_to_use.cpu()
    torch.cuda.empty_cache()
    
    return memory_used / (1024 * 1024)  # Convert to MB

def measure_inference_speed(model, tokenizer, dataset, batch_size=16, num_batches=10, token_map=None):
    """Measure average inference speed over multiple runs, with detailed timings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Wrap model with token mapping if provided
    wrapper = None
    if token_map is not None:
        wrapper = TokenMappingWrapper(model, token_map)
        model_to_use = wrapper
    else:
        model_to_use = model
    
    model_to_use = model_to_use.to(device)
    
    # Create DataLoader with subset of dataset
    def collate_fn(batch):
        # Extract text fields based on dataset format
        if 'sentence' in batch[0]:  # CoLA
            texts = [item['sentence'] for item in batch]
        elif 'sentence1' in batch[0]:  # MRPC, MNLI, etc.
            texts = [f"{item['sentence1']} {item['sentence2']}" for item in batch]
        else:
            texts = [str(item) for item in batch]  # Fallback
        
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Take a subset of the dataset for performance evaluation
    max_samples = min(batch_size * num_batches * 2, len(dataset))
    limited_dataset = dataset.select(range(max_samples))
    dataloader = DataLoader(limited_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Time for tokenization (measure on a batch)
    batch_iter = iter(dataloader)
    tokenize_start = time.time()
    inputs = next(batch_iter)
    tokenize_end = time.time()
    tokenize_time = tokenize_end - tokenize_start
    
    # Reset dataloader iterator
    batch_iter = iter(dataloader)
    
    # Move inputs to device - time this separately using the first batch
    first_batch = next(batch_iter)
    to_device_start = time.time()
    first_batch = {k: v.to(device) for k, v in first_batch.items()}
    to_device_end = time.time()
    to_device_time = to_device_end - to_device_start
    
    # Warmup with first batch
    with torch.no_grad():
        _ = model_to_use(**first_batch)
        # Reset mapping timing after warmup
        if wrapper is not None:
            wrapper.mapping_time = 0
            wrapper.mapping_calls = 0
    
    # Measure inference time on actual batches
    times = []
    dataloader_iter = iter(dataloader)
    batch_count = 0
    
    with torch.no_grad():
        try:
            while batch_count < num_batches:
                inputs = next(dataloader_iter)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                start_time = time.time()
                _ = model_to_use(**inputs)
                end_time = time.time()
                
                times.append(end_time - start_time)
                batch_count += 1
        except StopIteration:
            pass  # Reached end of dataset
    
    # Get mapping time if applicable
    mapping_time = 0
    if wrapper is not None:
        mapping_time = wrapper.mapping_time
        avg_mapping_time = wrapper.get_avg_mapping_time()
        mapping_percent = (avg_mapping_time / (sum(times) / len(times))) * 100 if times else 0
        print(f"Token mapping overhead: {avg_mapping_time*1000:.3f} ms per batch ({mapping_percent:.2f}% of inference time)")
    
    # Move model back to CPU
    model_to_use = model_to_use.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "tokenize_time": tokenize_time,
        "to_device_time": to_device_time,
        "total_mapping_time": mapping_time,
        "avg_mapping_time": mapping_time / len(times) if len(times) > 0 else 0,
        "num_batches_measured": len(times)
    }

def get_vocab_size_from_model(model_dir):
    """
    Determine the vocabulary size by examining the embedding weights.
    """
    # Check for pytorch_model.bin
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        # Load only the embeddings part
        state_dict = torch.load(model_path, map_location="cpu")
        for key in state_dict.keys():
            if "tok_embeddings.weight" in key or "token_embeddings.weight" in key or "word_embeddings.weight" in key:
                return state_dict[key].size(0)
    
    # Check for model.safetensors
    model_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(model_path):
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "tok_embeddings.weight" in key or "token_embeddings.weight" in key or "word_embeddings.weight" in key:
                        return f.get_tensor(key).size(0)
        except ImportError:
            print("safetensors not installed, skipping safetensors check")
    
    # Try to find config file
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            if "vocab_size" in config:
                return config["vocab_size"]
    
    return None

def create_pruned_model_directly(base_model_name, pruned_model_path, n_labels=2):
    """
    Create a pruned model directly with the correct structure
    based on weights from the pruned model.
    """
    print("Creating pruned model with correct structure...")
    
    # Get pruned vocabulary size
    pruned_vocab_size = get_vocab_size_from_model(pruned_model_path)
    if pruned_vocab_size is None:
        raise ValueError("Could not determine pruned vocabulary size")
    
    print(f"Detected pruned vocabulary size: {pruned_vocab_size}")
    
    # Load the base model to get the correct structure
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=n_labels)
    
    # Create a new embedding layer with the pruned vocabulary size
    embedding_dim = base_model.model.embeddings.tok_embeddings.embedding_dim
    new_embeddings = nn.Embedding(
        num_embeddings=pruned_vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=0  # Ensure padding_idx is set properly
    )
    
    # Replace the embedding layer
    base_model.model.embeddings.tok_embeddings = new_embeddings
    
    # Check for model file in different possible formats and locations
    possible_paths = [
        os.path.join(pruned_model_path, "pytorch_model.bin"),
        os.path.join(pruned_model_path, "model.bin"),
        os.path.join(pruned_model_path, "model.safetensors"),
        os.path.join(pruned_model_path, "..", "pytorch_model.bin"),
        pruned_model_path  # In case the full path to the model file was provided
    ]
    
    # Try to find a valid model file
    pruned_state_dict_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pruned_state_dict_path = path
            print(f"Found model file at: {path}")
            break
    
    if pruned_state_dict_path is None:
        # Instead of raising an error, print directories to help debug
        print(f"No state dict found at expected locations. Checking directory contents:")
        if os.path.exists(pruned_model_path):
            if os.path.isdir(pruned_model_path):
                print(f"Directory contents of {pruned_model_path}:")
                for item in os.listdir(pruned_model_path):
                    print(f"  - {item}")
            else:
                print(f"{pruned_model_path} exists but is not a directory")
        else:
            print(f"{pruned_model_path} does not exist")
            parent_dir = os.path.dirname(pruned_model_path)
            if os.path.exists(parent_dir):
                print(f"Parent directory {parent_dir} contents:")
                for item in os.listdir(parent_dir):
                    print(f"  - {item}")
        
        raise FileNotFoundError(f"Could not find model state dict for {pruned_model_path}")
    
    # Load the state dict
    if pruned_state_dict_path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            pruned_state_dict = {}
            with safe_open(pruned_state_dict_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    pruned_state_dict[key] = f.get_tensor(key)
        except ImportError:
            raise ImportError("safetensors not installed, but required to load this model")
    else:
        pruned_state_dict = torch.load(pruned_state_dict_path, map_location="cpu")
    
    # Load weights with strict=False to allow for embedding size mismatch
    base_model.load_state_dict(pruned_state_dict, strict=False)
    
    print("Successfully created pruned model with correct structure and loaded weights")
    return base_model

def load_dataset_for_task(task, split="test"):
    """Load the appropriate dataset for a given task."""
    print(f"Loading {task} dataset ({split} split)...")
    
    if task == "mrpc":
        dataset = load_dataset("glue", "mrpc", split=split)
    elif task == "mnli":
        dataset = load_dataset("glue", "mnli", split=split)
    elif task == "stsb":
        dataset = load_dataset("glue", "stsb", split=split)
    elif task == "cola":
        dataset = load_dataset("glue", "cola", split=split)
    elif task == "sst2":
        dataset = load_dataset("glue", "sst2", split=split)
    elif task == "qnli":
        dataset = load_dataset("glue", "qnli", split=split)
    elif task == "qqp":
        dataset = load_dataset("glue", "qqp", split=split)
    elif task == "rte":
        dataset = load_dataset("glue", "rte", split=split)
    elif task == "squad":
        dataset = load_dataset("squad", split=split)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    print(f"Loaded {len(dataset)} examples from {task} dataset")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Compare pruned ModernBERT model with base model")
    parser.add_argument("--pruned_model", type=str, required=True, help="Path to the pruned model")
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-base", 
                        help="Path to the base model (default: answerdotai/ModernBERT-base)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--task", type=str, default="mrpc", help="Task name (for dataset and number of labels)")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading base model for size measurement")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to use for GPU memory measurement")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to use for speed measurement")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use (test, validation, etc.)")
    args = parser.parse_args()
    
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    print("\n1. Loading models...")
    # Load base model and tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    
    # Load dataset
    dataset = load_dataset_for_task(args.task, args.dataset_split)
    
    # Get number of labels based on task
    n_labels = 2  # Default for many tasks
    if args.task == "mnli":
        n_labels = 3
    elif args.task == "stsb":
        n_labels = 1
    elif args.task == "cola":
        n_labels = 2
    
    # Try to determine token mapping
    token_map = None
    
    # Create pruned model with the correct structure
    try:
        pruned_model = create_pruned_model_directly(args.base_model, args.pruned_model, n_labels)
        
        # Extract token mapping from base and pruned models if not provided
        if token_map is None:
            # Create a basic mapping from original to pruned vocabulary
            # This is a simplified approach - a real implementation should 
            # extract the actual mapping from your pruning code
            pruned_vocab_size = pruned_model.model.embeddings.tok_embeddings.num_embeddings
            base_vocab_size = base_model.model.embeddings.tok_embeddings.num_embeddings
            
            # Create a simple mapping: map tokens to themselves if they're within range
            # or to UNK (0) if they're beyond the pruned vocab size
            token_map = {i: (i if i < pruned_vocab_size else 0) for i in range(base_vocab_size)}
            print(f"Created simple token mapping from {base_vocab_size} tokens to {pruned_vocab_size} tokens")
    except Exception as e:
        print(f"Error creating pruned model: {e}")
        return
    
    # Use base tokenizer for both models
    pruned_tokenizer = base_tokenizer
    
    print("\n2. Measuring model storage size...")
    # Try to determine the base model size
    base_size = None
    
    # Check if base model exists locally
    if os.path.exists(args.base_model) and os.path.isdir(args.base_model):
        base_size = get_model_size(args.base_model)
    else:
        # If not a local path, check if it's a downloaded Hugging Face model
        hf_cache_dir = os.environ.get("TRANSFORMERS_CACHE", 
                                     os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"))
        
        # Check common cache locations for Hugging Face models
        potential_cache_dirs = [
            hf_cache_dir,
            os.path.join(os.path.expanduser("~"), ".cache", "torch", "transformers"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        ]
        
        for cache_dir in potential_cache_dirs:
            if os.path.exists(cache_dir):
                # Look for model directory matching the base_model name
                model_id = args.base_model.replace("/", "--")
                for subdir in os.listdir(cache_dir):
                    if model_id in subdir and os.path.isdir(os.path.join(cache_dir, subdir)):
                        base_size = get_model_size(os.path.join(cache_dir, subdir))
                        print(f"Found base model in cache: {os.path.join(cache_dir, subdir)}")
                        break
            if base_size is not None:
                break
        
        # If we still don't have the base size, download it by default
        if base_size is None and not args.skip_download:
            print("Downloading base model to calculate exact size...")
            # Save the model locally to get its size
            temp_save_dir = "./temp_base_model"
            os.makedirs(temp_save_dir, exist_ok=True)
            base_model.save_pretrained(temp_save_dir)
            base_tokenizer.save_pretrained(temp_save_dir)
            base_size = get_model_size(temp_save_dir)
            print(f"Base model size: {base_size:.2f} MB")
            # Clean up (optional)
            # import shutil
            # shutil.rmtree(temp_save_dir)
            # print(f"Temporary files removed.")
    
    if base_size is None:
        base_params = sum(p.numel() for p in base_model.parameters())
        print(f"Base model parameters: {base_params:,}")
        print("Could not measure exact size. Use parameter count as a proxy.")
    else:
        print(f"Base model size: {base_size:.2f} MB")
    
    pruned_size = get_model_size(args.pruned_model)
    print(f"Pruned model size: {pruned_size:.2f} MB")
    
    if base_size:
        size_reduction = (1 - pruned_size / base_size) * 100
        print(f"Size reduction: {size_reduction:.2f}%")
    
    print("\n3. Comparing parameter count...")
    base_params = sum(p.numel() for p in base_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    print(f"Base model parameters: {base_params:,}")
    print(f"Pruned model parameters: {pruned_params:,}")
    param_reduction = (1 - pruned_params / base_params) * 100
    print(f"Parameter reduction: {param_reduction:.2f}%")
    
    # Check embedding size differences
    base_embedding_size = base_model.model.embeddings.tok_embeddings.weight.numel()
    pruned_embedding_size = pruned_model.model.embeddings.tok_embeddings.weight.numel()
    embedding_reduction = (1 - pruned_embedding_size / base_embedding_size) * 100
    print(f"Embedding parameters reduction: {embedding_reduction:.2f}%")
    
    print("\n4. Measuring GPU memory usage...")
    if torch.cuda.is_available():
        print(f"Running memory test with max {args.max_samples} samples and batch size {args.batch_size}")
        base_memory = measure_gpu_memory(base_model, base_tokenizer, dataset, 
                                        batch_size=args.batch_size, max_samples=args.max_samples)
        pruned_memory = measure_gpu_memory(pruned_model, pruned_tokenizer, dataset, 
                                          batch_size=args.batch_size, max_samples=args.max_samples, 
                                          token_map=token_map)
        
        print(f"Base model GPU memory: {base_memory:.2f} MB")
        print(f"Pruned model GPU memory: {pruned_memory:.2f} MB")
        memory_reduction = (1 - pruned_memory / base_memory) * 100
        print(f"Memory reduction: {memory_reduction:.2f}%")
    else:
        print("GPU not available. Skipping memory usage measurement.")
    
    print("\n5. Measuring inference speed...")
    print(f"Running speed test with {args.num_batches} batches and batch size {args.batch_size}")
    base_speed = measure_inference_speed(base_model, base_tokenizer, dataset, 
                                        batch_size=args.batch_size, num_batches=args.num_batches)
    pruned_speed = measure_inference_speed(pruned_model, pruned_tokenizer, dataset, 
                                          batch_size=args.batch_size, num_batches=args.num_batches, 
                                          token_map=token_map)
    
    print(f"Base model avg inference time: {base_speed['avg_time']*1000:.2f} ms (across {base_speed['num_batches_measured']} batches)")
    print(f"Pruned model avg inference time: {pruned_speed['avg_time']*1000:.2f} ms (across {pruned_speed['num_batches_measured']} batches)")
    speed_improvement = (1 - pruned_speed['avg_time'] / base_speed['avg_time']) * 100
    print(f"Speed improvement: {speed_improvement:.2f}%")
    
    print("\n6. Summary:")
    print(f"{'Metric':<20} {'Base Model':<15} {'Pruned Model':<15} {'Improvement':<10}")
    print("-" * 60)
    print(f"{'Parameters':<20} {base_params:,} {pruned_params:,} {param_reduction:.2f}%")
    print(f"{'Embedding Params':<20} {base_embedding_size:,} {pruned_embedding_size:,} {embedding_reduction:.2f}%")
    
    if base_size:
        print(f"{'Storage Size (MB)':<20} {base_size:.2f} {pruned_size:.2f} {size_reduction:.2f}%")
    
    if torch.cuda.is_available():
        print(f"{'GPU Memory (MB)':<20} {base_memory:.2f} {pruned_memory:.2f} {memory_reduction:.2f}%")
    
    print(f"{'Inference Time (ms)':<20} {base_speed['avg_time']*1000:.2f} {pruned_speed['avg_time']*1000:.2f} {speed_improvement:.2f}%")

if __name__ == "__main__":
    main()