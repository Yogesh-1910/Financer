from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    TextGenerationPipeline 
)
from peft import PeftModel
import torch
import os
import pandas as pd
import io 
from pathlib import Path 

# --- Configuration ---
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
# THIS MUST POINT TO YOUR ADAPTER FINE-TUNED ON INSTRUCTION DATA
# reflecting the behaviors described above.
adapter_model_path = "./phi3-finance-adapters/final_model_adapters"

device_is_cuda = torch.cuda.is_available()
if not device_is_cuda:
    print("WARNING: CUDA is not available. Inference will run on CPU and might be slow.")
print(f"Using device: {'cuda' if device_is_cuda else 'cpu'}")

# --- Global Variables ---
tokenizer = None
pipe = None
# Stop strings for post-processing. Lowercased for matching.
# These are for cleaning up *unexpected* model over-generation beyond its intended answer.
POST_PROCESS_STOP_STRINGS_LOWERCASE = [
    "### instruction:", "## instruction:", "# instruction:",
    "### response:", "## response:", "# response:",
    "human:", "user:",
    # Phrases the model might try to add if it doesn't follow the specific ending directive
    "is there anything else i can help you with?",
    "do you have any other questions?",
]
EOS_TOKEN_STR = "" 
SYSTEM_PROMPT = "You are FinBot, a specialized financial assistant. Your answers must be concise, accurate, and directly related to finance, investing, or budgeting. If a question is not about a financial topic, politely state that you can only assist with finance-related inquiries."

# --- Load Model and Tokenizer ---
try:
    if not os.path.isdir(adapter_model_path): 
        raise FileNotFoundError(f"Adapter model directory not found: {adapter_model_path}. Please ensure fine-tuning was successful and the path is correct.")

    print(f"Loading tokenizer. It's recommended to load from the adapter path if saved there, or base model if not.")
    # If you saved the tokenizer with your adapter (good practice, especially if special tokens were added/changed):
    # tokenizer = AutoTokenizer.from_pretrained(adapter_model_path, trust_remote_code=True)
    # If using base tokenizer (ensure it matches what was used for fine-tuning):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)


    if tokenizer.pad_token is None: 
        print("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    elif tokenizer.pad_token_id is None and tokenizer.pad_token: 
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    EOS_TOKEN_STR = tokenizer.eos_token
    if EOS_TOKEN_STR: 
        POST_PROCESS_STOP_STRINGS_LOWERCASE.append(EOS_TOKEN_STR.lower())
    POST_PROCESS_STOP_STRINGS_LOWERCASE = list(set(POST_PROCESS_STOP_STRINGS_LOWERCASE))

    bnb_config_inference = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if device_is_cuda and torch.cuda.is_bf16_supported() else torch.float16
    )
    print(f"Loading base model '{base_model_id}' with quantization...")
    base_model_for_peft = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config_inference,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Loading PEFT (LoRA) adapter from: {adapter_model_path}")
    model = PeftModel.from_pretrained(base_model_for_peft, adapter_model_path)
    model.eval() 

    print("Creating text-generation pipeline...")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    if isinstance(pipe, TextGenerationPipeline): print(f"Pipeline initialized. Model is on device: {pipe.device}")
    else: print("Pipeline initialized, but type is not TextGenerationPipeline.")
    print("Instruction-following model and pipeline loaded successfully.")

except FileNotFoundError as e: print(f"MODEL LOADING ERROR: FileNotFoundError - {e}"); pipe = None
except Exception as e: print(f"MODEL LOADING UNEXPECTED ERROR: {e}"); import traceback; traceback.print_exc(); pipe = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# In fastapi_server.py, replace the entire `convert_excel_to_text_summary` function with this new version.

def convert_excel_to_text_summary(df: pd.DataFrame) -> str:
    """
    Parses a budget DataFrame to create a clear text summary for the LLM.
    This version is specifically adapted to the provided Excel format.
    """
    print("Starting Excel to text summary conversion...")
    # Make a copy to avoid modifying the original DataFrame in place
    df = df.copy()
    
    # Standardize column names for easier and case-insensitive access.
    # This handles variations like 'Planned Monthly (INR)' vs 'planned monthly (inr)'
    original_columns = list(df.columns)
    df.columns = [str(col).lower().strip().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '') for col in original_columns]
    print(f"Standardized columns to: {list(df.columns)}")

    # Define the standardized names for the columns we need
    category_col = 'category'
    item_col = 'item'
    monthly_amount_col = 'planned_monthly_inr'
    type_col = 'type'
    
    # Check if all essential columns exist after standardization
    required_cols = [category_col, item_col, monthly_amount_col, type_col]
    if not all(col in df.columns for col in required_cols):
        return (f"Could not extract summary. One or more required columns are missing. "
                f"Expected to find headers like: {required_cols}. Found: {list(df.columns)}. "
                f"Please ensure your Excel file has 'Category', 'Item', 'Type', and 'Planned Monthly (INR)' columns.")

    # --- Data Cleaning and Type Conversion ---
    # Clean up text data and convert the amount column to a numeric type,
    # treating any conversion errors as 0. This prevents the app from crashing on bad data.
    df[category_col] = df[category_col].astype(str).str.lower().str.strip()
    df[item_col] = df[item_col].astype(str).str.strip()
    df[type_col] = df[type_col].astype(str).str.lower().str.strip()
    df[monthly_amount_col] = pd.to_numeric(df[monthly_amount_col], errors='coerce').fillna(0)

    # --- Data Extraction using Pandas filtering (more robust than looping) ---

    # Get the income value directly. Assumes one 'income' row.
    income_rows = df[df[type_col] == 'income']
    total_income = income_rows[monthly_amount_col].sum()

    # Get all expense rows (where Type is 'expense')
    expense_rows = df[df[type_col] == 'expense']
    total_expenses = expense_rows[monthly_amount_col].sum()

    # Calculate net savings
    net_savings = total_income - total_expenses
    
    # --- Build the Detailed Context for the LLM ---
    # This string will be fed into the AI to generate suggestions.

    context_for_llm = "Here is a summary of the user's monthly budget:\n"
    context_for_llm += f"- Total Monthly Income: {total_income:,.0f} INR\n"
    context_for_llm += f"- Total Monthly Expenses: {total_expenses:,.0f} INR\n"
    context_for_llm += f"- Net Monthly Savings: {net_savings:,.0f} INR\n\n"
    
    # Get the top 5 largest expense items to provide more specific advice
    top_5_expenses = expense_rows.sort_values(by=monthly_amount_col, ascending=False).head(5)

    if not top_5_expenses.empty:
        context_for_llm += "Here is a breakdown of their main expenses:\n"
        for _, row in top_5_expenses.iterrows():
            context_for_llm += f"- {row[item_col].capitalize()} ({row[category_col].capitalize()}): {row[monthly_amount_col]:,.0f} INR\n"
    else:
        context_for_llm += "No specific expense items were found.\n"

    # --- Final Check ---
    if total_income == 0 and total_expenses == 0:
        return "Could not find any valid income or expense entries in the uploaded Excel file. Please check the 'Type' and 'Planned Monthly (INR)' columns."
    
    print("Successfully generated detailed text summary from Excel for the LLM.")
    return context_for_llm # This is the detailed string that the /analyze_excel_and_advise endpoint will use in its prompt


# In fastapi_server.py, replace only this function

@app.post("/analyze_excel_and_advise")
async def analyze_excel_for_advice(file: UploadFile = File(...)):
    global pipe, tokenizer
    if not pipe or not tokenizer:
        raise HTTPException(status_code=503, detail="LLM not available.")
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        excel_df = pd.read_excel(io.BytesIO(contents))
        financial_summary_for_llm = convert_excel_to_text_summary(excel_df)
        
        if "Could not extract" in financial_summary_for_llm:
            raise HTTPException(status_code=400, detail=financial_summary_for_llm)
            
        summary_for_user_ui = financial_summary_for_llm.split("\n\nHere is a breakdown")[0].strip()

    except Exception as e:
        print(f"Excel Reading/Parsing Error: {e}")
        error_detail = str(e) if isinstance(e, ValueError) else "Could not read or process the Excel file. Please ensure it is a valid, uncorrupted Excel document."
        raise HTTPException(status_code=400, detail=error_detail)
    
    # --- PHI-3 PROMPT CONSTRUCTION ---
    # Create the user instruction for the model
    user_instruction = (
        "Based on the following financial summary, provide 3 to 5 concise, actionable financial advice points as a numbered or bulleted list. "
        "Prioritize the most impactful suggestions first. The advice should be clear and easy for a beginner to understand.\n\n"
        f"Financial Summary:\n{financial_summary_for_llm}"
    )

    # Apply the chat template just like in training
    messages = [{"role": "user", "content": user_instruction}]
    full_prompt_for_model = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # --- LLM INFERENCE ---
    try:
        print(f"DEBUG: Prompt (Excel ADVICE):\n'''{full_prompt_for_model}'''")
        
        generation_kwargs = {
            "max_new_tokens": 400,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # Phi-3 uses <|end|> as its EOS token. The pipeline should handle this automatically.
        results = pipe(full_prompt_for_model, **generation_kwargs)
        generated_text_full = results[0]['generated_text']
        
        # The response is the part after our prompt
        response_only = generated_text_full[len(full_prompt_for_model):].strip()

        # Clean up any trailing special tokens
        if response_only.endswith(tokenizer.eos_token):
            response_only = response_only[:-len(tokenizer.eos_token)].strip()
            
        print(f"DEBUG: Cleaned ADVICE from LLM:\n'''{response_only}'''")
        
        return {
            "filename_uploaded": file.filename,
            "financial_summary_sent_to_llm": summary_for_user_ui,
            "advice": response_only or "I'm sorry, I was unable to generate advice for that summary."
        }
    except Exception as e:
        print(f"ERROR during LLM inference (Excel ADVICE): {e}")
        raise HTTPException(status_code=500, detail=f"Error generating LLM advice: {str(e)}")


# In fastapi_server.py, replace the entire `ask_general_instruction` function

@app.post("/ask_general_instruction")
async def ask_general_instruction(request: Request):
    global pipe, tokenizer
    if not pipe or not tokenizer:
        raise HTTPException(status_code=503, detail="LLM not available.")
    try:
        data = await request.json()
        user_instruction_text = data.get('prompt')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    if not user_instruction_text or not isinstance(user_instruction_text, str) or not user_instruction_text.strip():
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")

    # --- PHI-3 PROMPT CONSTRUCTION WITH SYSTEM PROMPT ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction_text}
    ]
    full_prompt_for_model = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # The rest of the function remains the same...
    print(f"Prompt (General Q&A):\n'''{full_prompt_for_model}'''")
    try:
        generation_kwargs = {
            "max_new_tokens": 300,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        }
        results = pipe(full_prompt_for_model, **generation_kwargs)
        generated_text_full = results[0]['generated_text']
        
        # The response is the part after our prompt
        response_only = generated_text_full[len(full_prompt_for_model):].strip()
        
        # Clean up any trailing special tokens
        if response_only.endswith(tokenizer.eos_token):
            response_only = response_only[:-len(tokenizer.eos_token)].strip()

        # The system prompt is now part of the fine-tuning, so we don't need extensive logic here.
        # You can add simple keyword checks if you want an extra layer of topic guarding.
        
        print(f"Cleaned Response (General Q&A):\n'''{response_only}'''")
        return {"response": response_only}
    except Exception as e:
        print(f"ERROR (General Q&A): {e}")
        raise HTTPException(status_code=500, detail=f"Error in general instruction: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    if not pipe: print("CRITICAL: Model pipeline failed to load.")
    uvicorn.run(app, host="0.0.0.0", port=8000)