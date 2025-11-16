# main_api.py
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import pandas as pd
import io
import os
import traceback

# --- Model Configuration ---
MODEL_LOADED = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
adapter_path = os.path.join(SCRIPT_DIR, "phi2-finance-results-old-trl", "checkpoint-3") # Verify this path!

base_model_name = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None
system_prompt = "You are FinBot, a specialized financial assistant. Answer questions about personal finance, banking, investing, budgeting, and financial data analysis concisely and accurately. If a question is outside of these topics, politely state you can only assist with finance-related inquiries and nothing more."

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids): self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if stop_id in input_ids[0][-5:]: return True
        return False
stop_criteria = None

app = FastAPI(title="FinBot API", version="1.0.0")

# --- CORS, Pydantic Models ---
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel): message: str; history: list[ChatMessage] = []
class ChatResponse(BaseModel): reply: str
class ExcelAnalysisResponse(BaseModel): summary: str; suggestions: list[str]

def load_llm_model():
    global model, tokenizer, MODEL_LOADED, device, stop_criteria
    if MODEL_LOADED: return
    # ... (Model loading logic as before, ensure paths are correct)
    absolute_adapter_path = os.path.abspath(adapter_path)
    print(f"Attempting to load adapters from: {absolute_adapter_path}")
    if not os.path.isdir(adapter_path) or not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"\nCRITICAL ERROR: 'adapter_config.json' not found in path: {absolute_adapter_path}\n"); MODEL_LOADED = False; return
    
    print(f"Loading base model '{base_model_name}'...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    try:
        base_model_instance = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config if device == "cuda" else None, device_map="auto" if device == "cuda" else {"": "cpu"}, trust_remote_code=True, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        print("Loading PEFT adapters..."); model = PeftModel.from_pretrained(base_model_instance, adapter_path); model.eval()
        print("Loading tokenizer..."); tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        stop_token_ids = tokenizer.convert_tokens_to_ids(["[INST]"]); stop_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        MODEL_LOADED = True
        print("LLM Model and Tokenizer loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR loading LLM: {e}"); traceback.print_exc(); MODEL_LOADED = False

def clean_llm_reply(raw_reply_text):
    if "[INST]" in raw_reply_text: raw_reply_text = raw_reply_text.split("[INST]")[0]
    tokens_to_remove = ["<s>", "</s>", "<|endoftext|>", "FinBot:", "User:", "<<SYS>>", "<</SYS>>", "[/INST]"]
    cleaned_text = raw_reply_text
    for token in tokens_to_remove: cleaned_text = cleaned_text.replace(token, "")
    return cleaned_text.strip()

@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup..."); load_llm_model()
    if not MODEL_LOADED: print("\nCRITICAL WARNING: LLM Model failed to load.\n")

@app.post("/api/llm-chat", response_model=ChatResponse)
async def llm_chat_endpoint(request: ChatRequest):
    if not MODEL_LOADED: raise HTTPException(status_code=503, detail="LLM Model is not ready.")

    # --- THIS SECTION IS CORRECTED ---
    constructed_prompt_parts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    all_turns = request.history[-4:] + [ChatMessage(role="user", content=request.message)]
    is_first_user_content = True
    for turn in all_turns:
        role = turn.role # USE DOT NOTATION
        content = turn.content.strip() # USE DOT NOTATION
        if role == "user":
            if is_first_user_content:
                constructed_prompt_parts.append(f"{content} [/INST]")
                is_first_user_content = False
            else:
                constructed_prompt_parts.append(f"<s>[INST] {content} [/INST]")
        elif role == "assistant":
            constructed_prompt_parts.append(f" {content}</s>")

    full_prompt = "".join(constructed_prompt_parts)
    print(f"--- /api/llm-chat ---\nDEBUG: Final Prompt:\n{full_prompt}\n--------------------")

    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1800).to(model.device)
        generation_kwargs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "max_new_tokens": 350, "temperature": 0.6, "top_p": 0.9, "do_sample": True, "eos_token_id": tokenizer.eos_token_id, "stopping_criteria": stop_criteria}
        
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        
        raw_reply_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"DEBUG: Raw decoded LLM output:\n'{raw_reply_text}'\n--------------------")
        
        cleaned_reply = clean_llm_reply(raw_reply_text)
        print(f"DEBUG: Cleaned LLM output:\n'{cleaned_reply}'\n--------------------")
        
        if not cleaned_reply: cleaned_reply = "I'm not sure how to respond to that."
        return ChatResponse(reply=cleaned_reply)
    except Exception as e:
        print("ERROR in /api/llm-chat:"); traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error generating LLM reply.")

# ... (analyze_excel_endpoint remains the same as previous full code version)
# ... (__main__ block remains the same)
@app.post("/api/analyze-excel", response_model=ExcelAnalysisResponse)
async def analyze_excel_endpoint(file: UploadFile = File(...)):
    if not MODEL_LOADED: raise HTTPException(status_code=503, detail="LLM Model not loaded.")
    if not file.filename.endswith(('.xlsx', '.xls')): raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        contents = await file.read()
        excel_data = pd.read_excel(io.BytesIO(contents))
        required_cols = ['Category', 'Item', 'Planned Monthly (INR)', 'Type']
        missing_cols = [col for col in required_cols if col not in excel_data.columns]
        if missing_cols: raise ValueError(f"Missing required columns in Excel: {', '.join(missing_cols)}.")
        
        # --- Full Excel parsing logic here as before ---
        excel_summary_text = "..." # (Construct your text summary from excel_data here)
        # (This is a placeholder, your full parsing logic should be here)
        summary_for_response = "Summary text..." # placeholder
        
        prompt_for_llm = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nAnalyze... {excel_summary_text} ...[/INST]"
        
        inputs = tokenizer(prompt_for_llm, return_tensors="pt", max_length=1800, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, stopping_criteria=stop_criteria)
        
        raw_suggestions_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_suggestions = clean_llm_reply(raw_suggestions_text)
        suggestions_list = [s.strip() for s in cleaned_suggestions.split('\n') if s.strip()]

        return ExcelAnalysisResponse(summary=summary_for_response, suggestions=suggestions_list)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing Excel: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)