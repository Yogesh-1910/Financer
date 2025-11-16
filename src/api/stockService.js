// src/api/stockService.js
import axios from 'axios'; // Using axios for this example

// IMPORTANT: Replace with your Finnhub API Key
const FINNHUB_API_KEY = '#';      // YOUR_FINNHUB_API_KEY
const FINNHUB_BASE_URL = 'https://finnhub.io/api/v1';

// This constant determines if mock data is used.
const USE_MOCK_DATA = FINNHUB_API_KEY === 'YOUR_FINNHUB_API_KEY'; // Placeholder string
console.log("[stockService.js] FINNHUB_API_KEY:", FINNHUB_API_KEY);
console.log("[stockService.js] USE_MOCK_DATA (Finnhub):", USE_MOCK_DATA);

// Updated Mock data structure to somewhat resemble Finnhub's quote response
const MOCK_STOCKS_FINNHUB = {
  'IBM': { c: 168.50, d: 0.75, dp: 0.4500, h: 169.00, l: 167.50, o: 168.00, pc: 167.75, t: Date.now()/1000 },
  'AAPL': { c: 190.20, d: -1.10, dp: -0.5700, h: 191.00, l: 189.50, o: 190.50, pc: 191.30, t: Date.now()/1000 },
  'MSFT': { c: 425.80, d: 2.30, dp: 0.5420, h: 426.00, l: 423.50, o: 424.00, pc: 423.50, t: Date.now()/1000 },
  'GOOGL': { c: 170.10, d: 0.05, dp: 0.0290, h: 170.50, l: 169.80, o: 170.00, pc: 170.05, t: Date.now()/1000 },
};


export const getStockQuote = async (symbol) => {
  console.log(`[stockService.js] getStockQuote (Finnhub) called for symbol: ${symbol}`);
  const upperSymbol = symbol.toUpperCase();

  if (USE_MOCK_DATA) {
    console.warn(`[stockService.js] Using MOCK Finnhub stock data for ${upperSymbol}.`);
    return new Promise(resolve =>
      setTimeout(() => {
        // Finnhub quote endpoint returns the data directly, not nested
        const mockResponse = MOCK_STOCKS_FINNHUB[upperSymbol] || MOCK_STOCKS_FINNHUB['IBM'];
        if (mockResponse) {
            mockResponse.symbol = upperSymbol; // Add symbol for consistency if StockMonitor expects it
        }
        console.log(`[stockService.js] Returning MOCK Finnhub response for ${upperSymbol}:`, mockResponse);
        resolve(mockResponse);
      }, 500)
    );
  }

  const apiUrl = `${FINNHUB_BASE_URL}/quote?symbol=${upperSymbol}&token=${FINNHUB_API_KEY}`;
  console.log(`[stockService.js] Fetching LIVE Finnhub stock data from URL: ${apiUrl}`);

  try {
    const response = await axios.get(apiUrl); // Using axios.get
    console.log(`[stockService.js] Finnhub API response status for ${upperSymbol}: ${response.status}`);
    console.log(`[stockService.js] Finnhub API raw response data for ${upperSymbol}:`, response.data);

    // Finnhub returns data directly in response.data
    const quoteData = response.data;

    // Check for empty response or indicative error (Finnhub might return {} or specific error messages)
    // A successful quote usually has 'c' (current price) not equal to 0, unless the stock is truly at 0.
    if (Object.keys(quoteData).length === 0 || (quoteData.c === 0 && quoteData.pc === 0 && quoteData.t === 0) ) {
        // Finnhub might return an empty object for invalid symbols or if no data.
        // Or if "t" (timestamp) is 0, it often means no data.
        const errorMessage = `No data found for symbol ${upperSymbol} from Finnhub. It might be an invalid symbol or API issue.`;
        console.warn(`[stockService.js] ${errorMessage}. API response:`, quoteData);
        throw new Error(errorMessage);
    }
    
    // Add the symbol to the data if your StockMonitor component expects it at the top level
    // (Alpha Vantage put it inside 'Global Quote'.'01. symbol')
    quoteData.symbol = upperSymbol;

    console.log(`[stockService.js] Successfully returning Finnhub data for ${upperSymbol}:`, quoteData);
    return quoteData; // Return the quote data directly

  } catch (error) {
    let errorMessage;
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error(`[stockService.js] Finnhub API Error for ${upperSymbol} - Status: ${error.response.status}, Data:`, error.response.data);
      errorMessage = `Finnhub API Error for ${upperSymbol}: ${error.response.status}. ${error.response.data?.error || error.response.data || 'Unknown API error'}`;
    } else if (error.request) {
      // The request was made but no response was received
      console.error(`[stockService.js] Finnhub API No Response for ${upperSymbol}:`, error.request);
      errorMessage = `No response from Finnhub API for ${upperSymbol}. Check network or API status.`;
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error(`[stockService.js] Finnhub API Request Setup Error for ${upperSymbol}:`, error.message);
      errorMessage = `Error setting up Finnhub API request for ${upperSymbol}: ${error.message}`;
    }
    // Re-throw a new error with a consolidated message
    throw new Error(errorMessage);
  }
};