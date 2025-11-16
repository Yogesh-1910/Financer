// src/components/Dashboard/AIAssistant.js
import React, { useState, useEffect, useRef } from 'react';
import Card from '../UI/Card';
import Input from '../UI/Input';
import Button from '../UI/Button';
import styles from './AIAssistant.module.css';

import { sendMessageToFinBot, analyzeUploadedExcel } from '../../api/llmService';
import { speakText, startListening, stopListening } from '../../api/voiceService';

const AIAssistant = () => {
  const initialMessage =
    "Hello! I'm here to help you with any finance, investing, or budgeting questions you may have.";

  const [messages, setMessages] = useState([{ sender: 'ai', text: initialMessage }]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceError, setVoiceError] = useState('');

  const [selectedFile, setSelectedFile] = useState(null);
  const [fileError, setFileError] = useState('');
  const [excelAnalysisResult, setExcelAnalysisResult] = useState(null);
  const [isAnalyzingExcel, setIsAnalyzingExcel] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  // AUTO-SCROLL FIX
  useEffect(() => {
    requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages]);

  // Clean voice listener on exit
  useEffect(() => () => stopListening(), []);

  const escapeHTML = (txt = "") =>
    txt.replace(/&/g, "&amp;")
       .replace(/</g, "&lt;")
       .replace(/>/g, "&gt;");

  const handleSend = async (customText) => {
    const text = (customText ?? inputText).trim();
    if (!text || isLoading) return;

    setMessages(prev => [...prev, { sender: "user", text }]);
    if (!customText) setInputText("");

    setIsLoading(true);

    try {
      const aiText = await sendMessageToFinBot(text);
      setMessages(prev => [...prev, { sender: "ai", text: aiText }]);

      if (document.getElementById("speakAiResponse")?.checked) {
        speakText(aiText);
      }
    } catch (e) {
      setMessages(prev => [...prev, { sender: "ai", text: "Error: " + e.message }]);
    }

    setIsLoading(false);
  };

  const handleVoiceInput = async () => {
    if (isListening) {
      stopListening();
      setIsListening(false);
      return;
    }

    setIsListening(true);
    setVoiceError("");

    try {
      const transcript = await startListening();
      setInputText(prev => (prev + " " + transcript).trim());
    } catch (e) {
      setVoiceError(e.message);
    }

    setIsListening(false);
  };

  const handleFileChange = (e) => {
    setFileError('');
    const file = e.target.files?.[0];

    if (!file) return;

    if (!/\.(xlsx|xls)$/i.test(file.name)) {
      setFileError("Invalid file type (.xlsx / .xls only)");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setFileError("File too large (max 5MB)");
      return;
    }

    setSelectedFile(file);
    setExcelAnalysisResult(null);
  };

  const handleExcelUpload = async () => {
    if (!selectedFile) {
      setFileError("Please select a file first");
      return;
    }

    setIsAnalyzingExcel(true);

    try {
      const result = await analyzeUploadedExcel(selectedFile);
      setExcelAnalysisResult(result);
    } catch (e) {
      setExcelAnalysisResult({ error: e.message });
    }

    setIsAnalyzingExcel(false);
  };

  return (
    <Card title="AI Fin Bot" className={styles.AIWrapper}>

      {/* CHAT SECTION */}
      <div className={styles.chatContainer}>
        <div className={styles.chatWindow}>
          {messages.map((msg, i) => (
            <div key={i} className={`${styles.bubble} ${styles[msg.sender]}`}>
              <p dangerouslySetInnerHTML={{
                __html: escapeHTML(msg.text).replace(/\n/g, "<br/>")
              }} />
            </div>
          ))}

          {isLoading && (
            <div className={`${styles.bubble} ${styles.ai}`}>
              <i>FinBot is thinking...</i>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* INPUT BAR */}
        <div className={styles.inputBar}>
          <Input
            ref={inputRef}
            name="ai-input"
            type="text"
            placeholder="Ask a financial question..."
            value={inputText}
            disabled={isLoading}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
          />

          <Button onClick={() => handleSend()} disabled={!inputText.trim() || isLoading}>
            ➤
          </Button>

          <Button onClick={handleVoiceInput} variant={isListening ? "danger" : "secondary"}>
            {isListening ? <span className={styles.listening}></span> : "🎤"}
          </Button>
        </div>

        <label className={styles.speakOption}>
          <input type="checkbox" id="speakAiResponse" /> Speak AI Responses
        </label>

        {voiceError && <p className={styles.voiceError}>{voiceError}</p>}
      </div>

      {/* EXCEL ANALYZER */}
      <div className={styles.excelSection}>

        <h3>Analyze Your Budget Excel</h3>

        <div className={styles.fileInputRow}>
          <input
            type="file"
            id="excelFile"
            accept=".xlsx,.xls"
            ref={fileInputRef}
            className={styles.fileInput}
            onChange={handleFileChange}
          />
          <label
            htmlFor="excelFile"            
            className={styles.fileLabel}>
            {selectedFile ? selectedFile.name : "Choose Excel File..."}
          </label>

          {selectedFile && (
            <button className={styles.clearFile} onClick={() => {
              setSelectedFile(null);
              fileInputRef.current.value = "";
            }}>✕</button>
          )}
        </div>

        {fileError && <p className={styles.fileError}>{fileError}</p>}

        <Button onClick={handleExcelUpload} disabled={!selectedFile || isAnalyzingExcel}>
          {isAnalyzingExcel ? "Analyzing..." : "Analyze Selected Excel"}
        </Button>

        {excelAnalysisResult && (
          <div className={styles.analysisResult}>
            {excelAnalysisResult.error ? (
              <p className={styles.errorBox}>Error: {excelAnalysisResult.error}</p>
            ) : (
              <>
                <h4>Budget Summary</h4>
                <pre className={styles.summary}>{excelAnalysisResult.summary}</pre>

                <h4>AI Suggestions</h4>
                <ul className={styles.suggestionList}>
                  {excelAnalysisResult.suggestions?.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>

      <p className={styles.footerNote}>
        FinBot uses local fine-tuned Phi-3 mini
      </p>
    </Card>
  );
};

export default AIAssistant;
