# fine_tune_phi2_finance.py (ADAPTED FOR OLDER, PROBLEMATIC TRL VERSION)
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import os

# 1. Configuration (same as before)
model_name = "microsoft/phi-2"
dataset_path = os.path.join(os.path.dirname(__file__), "finance_qa.json")
output_dir = "./phi2-finance-results-old-trl-final" # A new output dir
log_dir = "./phi2-finance-logs-old-trl-final"

SYSTEM_PROMPT = "You are FinBot, a specialized financial assistant. Your answers must be concise, accurate, and to the point. If a question is not about personal finance, banking, investing, or budgeting, politely state that you can only assist with finance-related inquiries and nothing more."

# 2. Load the RAW dataset
print(f"Loading raw dataset from: {dataset_path}")
try:
    raw_train_dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Raw dataset loaded. Number of examples: {len(raw_train_dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}"); exit()

# 3. Load Model and Tokenizer
print(f"Loading base model and tokenizer: {model_name}")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Use right-side padding for consistency

MAX_SEQ_LENGTH = 512 # Define max sequence length here for tokenization

# 4. **CRITICAL CHANGE**: Manually format AND tokenize the dataset before training
def create_prompt_and_tokenize(examples_batch):
    # This function will format the text and tokenize it, returning tokenized data
    # that the old SFTTrainer can use directly.
    full_texts = []
    for i in range(len(examples_batch['instruction'])):
        instruction = examples_batch['instruction'][i]
        input_text = examples_batch['input'][i] if examples_batch.get('input') and examples_batch['input'][i] else ""
        response = examples_batch['output'][i]
        if input_text:
            text = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{instruction}\n{input_text} [/INST] {response} </s>"
        else:
            text = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{instruction} [/INST] {response} </s>"
        full_texts.append(text)

    # Tokenize the entire batch of formatted text strings
    # The tokenizer handles padding and truncation
    tokenized_output = tokenizer(
        full_texts,
        truncation=True,
        padding=False, # SFTTrainer might handle its own padding, False is safer here if unsure
                       # Or set to 'max_length' if you want uniform length
        max_length=MAX_SEQ_LENGTH,
    )
    # Important: The trainer will also need labels. For language modeling, labels are often a copy of input_ids.
    tokenized_output["labels"] = tokenized_output["input_ids"][:]
    return tokenized_output


print("Manually pre-formatting and tokenizing dataset...")
# We are creating the final, tokenized dataset that the old SFTTrainer can consume directly
tokenized_train_dataset = raw_train_dataset.map(
    create_prompt_and_tokenize,
    batched=True,
    remove_columns=raw_train_dataset.column_names # Remove all original columns ('instruction', 'input', 'output')
)
print("Dataset successfully tokenized.")
print(f"Columns in final dataset: {tokenized_train_dataset.column_names}")
# This should now print: ['input_ids', 'attention_mask', 'labels']


# 5. PEFT and Model Preparation
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

def print_trainable_parameters(model): # (same function as before)
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel();
        if param.requires_grad: trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")
print_trainable_parameters(model)


# 6. Training Arguments (remains largely the same)
training_args = TrainingArguments(
    output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=2e-4, logging_steps=50, # Log less frequently for a larger dataset
    num_train_epochs=3, # Train for 3 epochs on your 1000 examples
    save_strategy="epoch", optim="paged_adamw_8bit", lr_scheduler_type="cosine",
    bf16=True, report_to="tensorboard", logging_dir=log_dir,
)

# 7. Initialize SFTTrainer (with the old API signature)
print("Initializing SFTTrainer (using minimal, older API structure)...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, # <-- Pass the PRE-TOKENIZED dataset
    # NO: peft_config=...       (The model passed is already a PeftModel)
    # NO: dataset_text_field=... (Not a valid argument for your version)
    # NO: max_seq_length=...    (Handled during the manual tokenization step)
    # NO: tokenizer=...         (Not a valid argument for your version)
    # NO: formatting_func=...   (We already did this manually)
    # NO: packing=...           (Not a valid argument for your version)
)


# 8. Start Fine-tuning
print("Starting fine-tuning...")
try:
    trainer.train()
    print("Fine-tuning finished successfully.")
except Exception as e:
    print(f"Error during training: {e}"); import traceback; traceback.print_exc(); exit()


# 9. Save the Final Model Adapters
final_adapter_path = os.path.join(output_dir, "final_model_adapters")
try:
    print(f"Saving final adapters to {final_adapter_path}")
    trainer.model.save_pretrained(final_adapter_path)
    # Also save the tokenizer we used, for consistency in inference
    tokenizer.save_pretrained(final_adapter_path)
    print("Final adapters and tokenizer saved successfully.")
except Exception as e:
    print(f"Error saving model/tokenizer: {e}")

print("Script finished.")