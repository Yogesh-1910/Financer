// src/api/llmService.js

// This URL must point to your running FastAPI backend server.
const API_BASE_URL = 'http://localhost:8000'; // Note: No '/api' prefix, matching your new Python code

/**
 * Sends a chat message to the "/ask_general_instruction" endpoint.
 * This endpoint, as designed in your new python file, is stateless and does not use chat history.
 * @param {string} message - The current user message, which will be the 'prompt' for the backend.
 * @returns {Promise<string>} - The AI's text response.
 * @throws {Error}
 */
export const sendMessageToFinBot = async (message) => { // chatHistory parameter is removed
  const payload = {
    prompt: message,
  };

  console.log("Frontend Service: Sending payload to /ask_general_instruction:", payload);

  try {
    const response = await fetch(`${API_BASE_URL}/ask_general_instruction`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Server error" }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    const data = await response.json(); // Expected format: { "response": "..." }
    console.log("Frontend Service: Received from /ask_general_instruction:", data);

    if (typeof data.response !== 'string') {
      throw new Error("Invalid response format. Expected 'response' key.");
    }
    
    return data.response;

  } catch (error) {
    console.error('Frontend Service: Error communicating with FinBot API:', error.message);
    if (error.message.includes('Failed to fetch')) {
        throw new Error(`Error: Could not connect to FinBot. Ensure backend is running at ${API_BASE_URL}.`);
    }
    throw error;
  }
};

/**
 * Uploads an Excel file to the "/analyze_excel_and_advise" endpoint.
 * @param {File} file - The Excel file object.
 * @returns {Promise<Object>} - An object with { summary, suggestions }
 * @throws {Error}
 */
export const analyzeUploadedExcel = async (file) => {
  const formData = new FormData();
  formData.append("file", file); // The key "file" must match `file: UploadFile = File(...)` in Python

  console.log(`Frontend Service: Sending Excel file to /analyze_excel_and_advise`);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze_excel_and_advise`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Excel analysis failed." }));
      throw new Error(errorData.detail || `Excel analysis failed with status ${response.status}`);
    }

    const data = await response.json(); // Expected format from Python: { advice: "...", financial_summary_sent_to_llm: "...", ... }
    console.log("Frontend Service: Received from /analyze_excel_and_advise:", data);

    // Transform the backend response to the format the AIAssistant component expects
    return {
        summary: data.financial_summary_sent_to_llm || "No summary was generated.",
        // The backend sends 'advice' as a single string paragraph. We wrap it in an array for consistent handling.
        suggestions: data.advice ? [data.advice] : []
    };
  } catch (error) {
    console.error('Frontend Service: Error analyzing Excel via API:', error.message);
    throw error;
  }
};