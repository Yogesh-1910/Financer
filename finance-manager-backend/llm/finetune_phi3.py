import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer
import os

# --- Configuration ---
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
TRAIN_DATASET_PATH = "finance_train.json"
VALID_DATASET_PATH = "finance_validation.json"
OUTPUT_DIR = "./phi3-finance-adapters"
LOG_DIR = "./logs/phi3-finetune-logs"
tokenizer = None # Global tokenizer

SYSTEM_PROMPT = "You are FinBot, a specialized financial assistant. Your answers must be concise, accurate, and directly related to finance, investing, or budgeting. If a question is not about a financial topic, politely state that you can only assist with finance-related inquiries."

def format_chat_template(example):
    """
    Applies the Phi-3 chat template WITH A SYSTEM PROMPT.
    """
    return {
        "text": tokenizer.apply_chat_template(
            [
                # Add the system prompt here to define the model's persona
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ],
            tokenize=False,
        )
    }


def main():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading and preparing datasets...")
    train_dataset = load_dataset("json", data_files=TRAIN_DATASET_PATH, split="train").map(format_chat_template)
    validation_dataset = load_dataset("json", data_files=VALID_DATASET_PATH, split="train").map(format_chat_template)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model '{MODEL_ID}'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager", # Keep "eager" for Windows
        torch_dtype=compute_dtype,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules='all-linear',
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,      # KEEP at 1
        gradient_accumulation_steps=8,      # KEEP at 8
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir=LOG_DIR,
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=100,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        bf16=is_bf16_supported,
        fp16=not is_bf16_supported,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        # --- MEMORY SAVING CHANGES ---
        max_seq_length=512,  # Reduce sequence length further
        packing=False,       # Disable packing
        # --- END CHANGES ---
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    final_save_path = os.path.join(OUTPUT_DIR, "final_model_adapters")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Final LoRA adapter and tokenizer saved to '{final_save_path}'")

if __name__ == "__main__":
    main()