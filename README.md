# 🚀 Professional Crypto Analytics Platform

A comprehensive, real-time cryptocurrency dashboard built with Python and Streamlit. This project showcases a full-stack data application, offering advanced analytics, AI-powered insights, and a personal portfolio management system. It's an ideal tool for serious crypto enthusiasts and traders who need a powerful, all-in-one platform for market analysis.

<p align="center">
<a href="https://www.linkedin.com/in/hitanshgopani/">
<img src="https://img.shields.io/badge/Connect%20on%20LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn Profile">
</a>
<a href="#">
<img src="https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python" alt="Python">
</a>
<a href="#">
<img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
</a>
</p>

## ✨ Key Features

* **Real-Time Data**: Live data for the top cryptocurrencies, fetched and auto-refreshed from the CoinMarketCap API.

* **AI-Powered Analysis**: A dedicated section to ask questions to an AI market analyst powered by the **Gemini API**.

* **Advanced Technical Analysis**: Interactive charts with multiple views and key indicators like Moving Averages, Bollinger Bands, RSI, and MACD.

* **Interactive Market Screener**: A powerful tool to filter and find cryptocurrencies based on custom criteria.

* **Portfolio Management**: Track your crypto holdings, calculate P&L, and monitor your total portfolio value.

* **Market Insights**: Visualize market cap distribution, top movers, and a real-time Fear & Greed Index.

## ⚙️ How to Run Locally

To get a copy of this project running on your local machine, follow these steps.

### Prerequisites

* Python 3.9 or higher

* `git` installed on your system

### Installation

1.  **Clone the repository**:
    ```
    git clone [https://github.com/Professional50coder/Dashboard_crypto.git](https://github.com/Professional50coder/Dashboard_crypto.git)
    cd Dashboard_crypto
    ```

2.  **Create and activate a virtual environment**:
    ```
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages**:
    ```
    pip install -r requirements.txt
    ```

### API Keys and Configuration

This project requires API keys. Follow the official Streamlit guide for [Secrets Management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

1.  Create a new directory named `.streamlit` in the root of your project.

2.  Inside `.streamlit`, create a file named `secrets.toml`.

3.  Add your API keys and URLs to `secrets.toml`:
    ```toml
    [api_keys]
    coinmarketcap = "YOUR_COINMARKETCAP_API_KEY"
    gemini = "YOUR_GEMINI_API_KEY"

    [api_urls]
    cmc_latest_listings = "[https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest](https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest)"
    cmc_ohlcv_historical = "[https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical](https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical)"
    cmc_global_metrics = "[https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest](https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest)"
    ```
    *Note: For the best experience, use a live production key for CoinMarketCap. The sandbox key `b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c` can be used for testing, but historical data will be mocked.*

### Running the Application

Once everything is configured, run the dashboard from your terminal:

Your browser will automatically open to `http://localhost:8501`.

## ☁️ Deployment

This project is optimized for deployment on **Streamlit Cloud**. To deploy, connect your GitHub repository and provide the contents of your `secrets.toml` file via the app's advanced settings.

---

### **Connect with the Author**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/hitanshgopani/)
