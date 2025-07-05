# Financier - AI-Powered Personal Finance Manager

Financier is a modern, full-stack web application designed to help users manage their finances effectively. It features a responsive React frontend and a powerful Python backend powered by a fine-tuned Phi-2 large language model (LLM). The application provides tools for budget tracking, loan management, stock monitoring, and an intelligent financial assistant named FinBot.

 <!-- Replace with a good screenshot of your app's dashboard -->

## ✨ Features

-   **Secure Authentication:** User sign-up and login system.
-   **Dashboard Overview:** A central hub displaying user profile details and the latest financial news.
-   **Budget Manager:** An Excel-like interface to track monthly income and expenses, with automatic calculations for savings.
-   **Loan & EMI Tracker:** A tool to monitor loans and Equated Monthly Installments (EMIs).
-   **Stock Monitor:** Real-time stock price tracking with interactive charts for selected symbols (powered by Finnhub API).
-   **🤖 AI FinBot:**
    -   A specialized financial chatbot powered by a fine-tuned Microsoft Phi-2 model.
    -   Answers general financial questions concisely and accurately.
    -   Refuses to answer non-financial queries.
    -   **Excel Budget Analyzer:** Users can upload their budget spreadsheet, and FinBot will provide a summary and actionable financial advice.
-   **🎤 Voice Assistant:** Supports voice commands and can speak AI responses (using Web Speech API).
-   **Responsive Design:** Clean and professional UI that works on both desktop and mobile devices.

---

## 🛠️ Tech Stack

**Frontend:**
-   **React:** A JavaScript library for building user interfaces.
-   **React Router:** For client-side routing.
-   **CSS Modules:** For component-scoped styling.
-   **Chart.js / react-chartjs-2:** For rendering stock charts.

**Backend (LLM & API):**
-   **Python:** The core language for the backend.
-   **FastAPI:** A modern, high-performance web framework for building APIs.
-   **PyTorch:** The deep learning framework used for the LLM.
-   **Hugging Face Transformers:** For loading and working with the Phi-2 model.
-   **Hugging Face PEFT (LoRA/QLoRA):** For parameter-efficient fine-tuning.
-   **Hugging Face TRL:** For supervised fine-tuning (`SFTTrainer`).
-   **Pandas & Openpyxl:** For parsing and analyzing uploaded Excel files.
-   **Uvicorn:** An ASGI server for running FastAPI.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally. You will need to run the backend server and the frontend application in separate terminal windows.

### Prerequisites

-   **Node.js and npm** (or yarn) for the frontend.
-   **Python 3.8+** and `pip` for the backend.
-   **Git** for cloning the repository.
-   An **NVIDIA GPU with CUDA** is highly recommended for running the LLM backend.

### Backend Setup (LLM & API Server)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/financier.git
    cd financier
    ```

2.  **Navigate to the backend directory:**
    ```bash
    cd finance-manager-backend/llm # Adjust path as per your structure
    ```

3.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv_backend
    # On Windows:
    venv_backend\Scripts\activate
    # On macOS/Linux:
    source venv_backend/bin/activate
    ```

4.  **Install Python dependencies:**
    *It's recommended to install PyTorch first, matching your system's CUDA version.*
    ```bash
    # Example for CUDA 11.8. Check PyTorch website for your version.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install other dependencies
    pip install transformers datasets accelerate bitsandbytes sentencepiece peft trl
    pip install fastapi uvicorn python-multipart pandas openpyxl
    pip install tensorboard # For monitoring training
    ```

5.  **Fine-tune the Phi-2 Model (One-time setup):**
    *   **Prepare your dataset:** Create a `finance_qa.json` file in the `llm` directory. This file should contain a list of instruction-output pairs for training the model on financial Q&A.
    *   **Run the fine-tuning script:**
        ```bash
        python fine_tune_phi2_finance.py
        ```
    *   This will create a folder (e.g., `phi2-finance-results/final_phi2_finance_adapters`) containing the fine-tuned LoRA adapter weights. Ensure the `adapter_model_path` in `main_api.py` points to this directory.

6.  **Start the Backend API Server:**
    ```bash
    uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
    ```
    *   The backend server should now be running. Check the terminal for "LLM Model and Tokenizer loaded successfully." and "Uvicorn running on http://0.0.0.0:8000".

### Frontend Setup

1.  **Open a new terminal window.**

2.  **Navigate to the frontend project root:**
    ```bash
    cd path/to/your/project/ # The folder containing src, public, package.json
    ```

3.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

4.  **Set up API Keys:**
    *   **Stock API:** Get a free API key from [Finnhub.io](https://finnhub.io/dashboard). Open `src/api/stockService.js` and replace `'YOUR_FINNHUB_API_KEY'` with your actual key.
    *   **Financial News API:** Get a free API key from [NewsAPI.org](https://newsapi.org/). Open `src/api/newsService.js` and replace `'YOUR_NEWS_API_KEY'` with your key.

5.  **Start the React Development Server:**
    ```bash
    npm start
    ```

6.  **Access the Application:**
    *   Open your browser and navigate to `http://localhost:3000`. You should now see the Financier login page.

---

## 📝 Usage

-   **Sign Up / Login:** Create an account or log in to access the dashboard.
-   **AI FinBot:**
    -   Ask general financial questions in the chat input.
    -   Click "Analyze Your Budget Excel", choose an Excel file with your budget details, and click "Analyze Selected Excel" to get a summary and AI-powered suggestions.
-   **Other Modules:** Navigate via the sidebar to access the Budget Manager, Loan & EMI Tracker, and Stock Monitor.

---

## Folder Structure

financier/
```
financier/
├── finance-manager-backend/
│   └── llm/
│       ├── fine_tune_phi2_finance.py    # Script for fine-tuning
│       ├── main_api.py                  # FastAPI backend server
│       ├── test_finetuned_model.py      # Script to test model locally
│       ├── finance_qa.json              # Your training dataset
│       └── phi2-finance-results/        # Output directory for adapters
├── src/                                 # React frontend source
│   ├── api/                             # API service calls (LLM, Stocks, News)
│   ├── components/                      # Reusable React components
│   ├── contexts/                        # React contexts (e.g., AuthContext)
│   ├── pages/                           # Page-level components
│   ├── App.js                           # Main app component and router
│   └── ...
├── public/
├── package.json
└── README.md
```

## 💻Developers
**Yogesh S** -
https://github.com/Yogesh-1910

**Danush G** - 
https://github.com/Danush6123

**Hemanth P** -
https://github.com/Hemanth-0013
