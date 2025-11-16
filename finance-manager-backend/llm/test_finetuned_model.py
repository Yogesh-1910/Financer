# test_finetuned_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

base_model_name = "microsoft/phi-2"
adapter_path = "./phi2-finance-results-old-trl/checkpoint-3" # Path where your adapters were saved
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load the LoRA adapters
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device) # Ensure model is on the correct device
model.eval() # Set to evaluation mode

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# System prompt for your FinBot
system_prompt = "You are FinBot, a specialized financial assistant. Answer concisely and accurately. If a question is not about personal finance, banking, investing, or budgeting, politely state you only answer finance-related questions."

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Basic prompt template (ChatML-like or instruction-based)
    # Adjust this to match the format used during fine-tuning
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

    print("FinBot is thinking...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response_ids = outputs[0][inputs["input_ids"].shape[1]:] # Get only the generated tokens
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    print(f"FinBot: {response_text.strip()}")