// src/api/newsService.js
import axios from 'axios';

// Load API Key
const NEWS_API_KEY = process.env.REACT_APP_NEWS_API_KEY;

// Check if API key exists
const IS_NEWS_API_KEY_CONFIGURED_FOR_LIVE =
  NEWS_API_KEY &&
  NEWS_API_KEY.trim() !== '' &&
  NEWS_API_KEY !== 'YOUR_NEWS_API_KEY' &&
  NEWS_API_KEY !== 'YOUR_API_KEY';

if (!IS_NEWS_API_KEY_CONFIGURED_FOR_LIVE) {
  console.warn(
    `[NewsService] WARNING: REACT_APP_NEWS_API_KEY is missing or invalid.
    Current Key Prefix: "${NEWS_API_KEY ? NEWS_API_KEY.substring(0, 10) + '...' : 'Not Set'}"
    Using MOCK data fallback.`
  );
}

// ----------------------
// MOCK NEWS DATA
// ----------------------
const MOCK_NEWS_DATA_ARTICLES = [
  {
    source: { id: null, name: "Mock Finance Today" },
    author: "AI Reporter",
    title: "Market Trends: A Look Ahead (Mock Data)",
    description: "Analysts discuss potential market movements for the upcoming quarter. Mock data only.",
    url: "#mock-article-1",
    urlToImage: "https://via.placeholder.com/120x80.png?text=MarketTrends",
    publishedAt: new Date(Date.now() - 3600 * 1000).toISOString(),
  },
  {
    source: { id: null, name: "Mock Business Insights" },
    author: "Demo Journalist",
    title: "Impact of New Policies on Small Businesses (Mock Data)",
    description: "Exploring how regulatory changes affect small enterprises. Mock data only.",
    url: "#mock-news-2",
    urlToImage: "https://via.placeholder.com/120x80.png?text=SMBPolicy",
    publishedAt: new Date(Date.now() - 4 * 3600 * 1000).toISOString(),
  },
  {
    source: { id: null, name: "Mock Global Finance" },
    author: "Economic Bot",
    title: "Currency Fluctuations and Global Trade (Mock Data)",
    description: "An overview of currency movements and global trade. Mock data only.",
    url: "#mock-news-3",
    urlToImage: "https://via.placeholder.com/120x80.png?text=CurrencyNews",
    publishedAt: new Date(Date.now() - 7 * 3600 * 1000).toISOString(),
  }
];

// ----------------------
// MAIN FUNCTION
// ----------------------
export const getFinanceNews = async () => {

  // Return mock data if key missing
  if (!IS_NEWS_API_KEY_CONFIGURED_FOR_LIVE) {
    console.log("[NewsService] Returning MOCK news due to missing API key.");
    return new Promise(resolve => setTimeout(() => resolve(MOCK_NEWS_DATA_ARTICLES), 200));
  }

  const country = 'us'; // 'in' often returns 0 results. Using 'us' for consistent data.
  const pageSize = 5;

  const TOP_HEADLINES_URL = 
    `https://newsapi.org/v2/top-headlines?country=${country}&category=business&pageSize=${pageSize}&apiKey=${NEWS_API_KEY}`;

  const EVERYTHING_FALLBACK_URL =
    `https://newsapi.org/v2/everything?q=finance&language=en&pageSize=${pageSize}&sortBy=publishedAt&apiKey=${NEWS_API_KEY}`;

  try {
    // ----------------------------
    // STEP 1: Attempt Top Headlines
    // ----------------------------
    console.log("[NewsService] Fetching LIVE top-headlines:", TOP_HEADLINES_URL);

    const topResponse = await axios.get(TOP_HEADLINES_URL);
    console.log("[NewsService] Top-headlines status:", topResponse.status);

    // If top-headlines returned articles
    if (topResponse.data.status === "ok" && topResponse.data.articles.length > 0) {
      console.log("[NewsService] Got live headlines:", topResponse.data.articles.length);
      return topResponse.data.articles;
    }

    // If top-headlines returned 0 articles → fallback to everything endpoint
    console.warn("[NewsService] 0 articles from top-headlines. Trying EVERYTHING API...");

    const everythingResponse = await axios.get(EVERYTHING_FALLBACK_URL);

    if (everythingResponse.data.status === "ok" && everythingResponse.data.articles.length > 0) {
      console.log("[NewsService] EVERYTHING API returned:", everythingResponse.data.articles.length);
      return everythingResponse.data.articles;
    }

    // Everything API also failed → fallback mock
    console.warn("[NewsService] Everything API returned 0 articles. Using MOCK.");
    return MOCK_NEWS_DATA_ARTICLES;

  } catch (error) {
    console.error("[NewsService] Error fetching news:", error);
    console.warn("[NewsService] Using MOCK data due to error.");
    return MOCK_NEWS_DATA_ARTICLES;
  }
};
