import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os # For os.path.join
# from multiprocess import freeze_support # Good for Windows if freezing, optional here

# --- Global Variables / Functions (can be outside main guard) ---
model_id = "microsoft/phi-2"
dataset_file = "finance_qa.json"
output_dir = "./phi2-finance-lora-v1-ada"
log_dir = "./logs/phi2-finetune-logs"
tokenizer = None # Initialize tokenizer globally so format_example can access it

def format_example(example):
    global tokenizer # Access the global tokenizer
    instruction = example.get('instruction', '')
    output = example.get('output', '')
    eos = tokenizer.eos_token if tokenizer and tokenizer.eos_token else ''
    return {
        "text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}{eos}"
    }

def tokenize_function(examples):
    global tokenizer # Access the global tokenizer
    if "text" not in examples:
        print("Error: 'text' field not found in examples after formatting.")
        return {}
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )

# --- Main Execution Block ---
def main():
    global tokenizer # Allow main to modify the global tokenizer

    # --- Check for CUDA and BF16 support ---
    is_cuda_available = torch.cuda.is_available()
    is_bf16_supported = is_cuda_available and torch.cuda.is_bf16_supported()

    if not is_cuda_available:
        print("WARNING: CUDA is not available. Training will run on CPU...")
    
    # --- Load tokenizer ---
    try:
        print(f"Loading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer with use_fast=True: {e}. Trying use_fast=False...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        except Exception as e2:
            print(f"Still failed to load tokenizer: {e2}")
            return # Exit main function

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None : tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            print("ERROR: Tokenizer has no pad_token and no eos_token.")
            return

    # --- Load and prepare dataset ---
    try:
        print(f"Loading dataset from '{dataset_file}'...")
        dataset = load_dataset("json", data_files=dataset_file, split="train")
        dataset = dataset.map(format_example, remove_columns=list(dataset.column_names))
        
        print("Tokenizing dataset...")
        # OPTION 1 (SAFEST FOR WINDOWS to avoid multiprocessing RuntimeError):
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # OPTION 2 (If you want to try multiprocessing, ensure this script is run with if __name__ == '__main__'):
        # tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=2) # num_proc > 1 needs the __main__ guard
        print("Dataset tokenized.")
    except Exception as e:
        print(f"Error during dataset loading or processing: {e}")
        import traceback; traceback.print_exc()
        return

    # --- Configure BitsAndBytes for 4-bit quantization ---
    bnb_compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16
    print(f"Using bnb_compute_dtype: {bnb_compute_dtype}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_compute_dtype
    )

    # --- Load model ---
    print(f"Loading base model '{model_id}' with quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",  # KEEPING THIS - ASSUMING LIBRARY UPGRADE IS THE FIX
            trust_remote_code=True,
            torch_dtype=bnb_compute_dtype
        )
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"ERROR LOADING MODEL: {e}")
        print("This is likely the 'TypeError: argument of type 'NoneType' is not iterable'.")
        print("Ensure 'accelerate', 'transformers', 'bitsandbytes' are UP-TO-DATE.")
        print("If error persists, try commenting out 'device_map=\"auto\"' and adding 'model = model.to(\"cuda\")' after this block.")
        import traceback; traceback.print_exc()
        return

    # --- Prepare model for QLoRA ---
    try:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training.")
    except Exception as e:
        print(f"Error preparing model for k-bit training: {e}")
        return

    # --- Apply LoRA ---
    target_modules_phi2 = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"] # Adjust if needed
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules_phi2,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    try:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as e:
        print(f"Error applying LoRA config: {e}")
        return

    # --- Training config ---
    training_args_dict = {
        "output_dir": output_dir, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8, "num_train_epochs": 3,
        "learning_rate": 2e-4, "logging_dir": log_dir,
        "logging_strategy": "steps", "logging_steps": 10,
        "save_strategy": "epoch", "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "report_to": "tensorboard", "ddp_find_unused_parameters": False,
    }
    if is_cuda_available:
        training_args_dict["bf16"] = is_bf16_supported
        training_args_dict["fp16"] = not is_bf16_supported
    training_args = TrainingArguments(**training_args_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_datasets,
        data_collator=data_collator
    )

    # --- Train model ---
    print("Starting training...")
    try:
        train_result = trainer.train()
        print("Training complete.")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        final_save_path = os.path.join(output_dir, "final_adapter")
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"Final LoRA adapter and tokenizer saved to {final_save_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback; traceback.print_exc()

    print("Fine-tuning script finished.")

if __name__ == '__main__':
    # freeze_support() # Good for Windows if creating executables, optional for scripts
    main()