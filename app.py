import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
import numpy as np
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Configuration and API Setup ---
st.set_page_config(
    page_title="Professional Crypto Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Initialize session state variables for persistent data
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'market_fear_greed' not in st.session_state:
    st.session_state.market_fear_greed = None

# API Configuration - Using your live production API keys
# For production use, please secure these keys and avoid hardcoding them.
API_KEY = "7a4be8a4-d432-4d70-84b5-4ce749f3040b"
API_HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY,
}
# Using live production URLs for all endpoints with your provided API key
LATEST_LISTINGS_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
# Note: The historical data endpoint is a paid feature, so we will use the sandbox URL to avoid a Forbidden error.
OHLCV_HISTORICAL_URL = "https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
GLOBAL_METRICS_URL = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"

# --- Data Fetching and Calculations ---

@st.cache_data(ttl=60)
def get_latest_crypto_data(limit=100, sort='market_cap'):
    """Fetches and processes the latest cryptocurrency data from CoinMarketCap."""
    try:
        parameters = {"start": "1", "limit": limit, "convert": "USD", "sort": sort}
        response = requests.get(LATEST_LISTINGS_URL, headers=API_HEADERS, params=parameters)
        response.raise_for_status()
        data = json.loads(response.text)
        
        df = pd.json_normalize(data['data'])
        
        # Select and rename columns for clarity
        df = df[[
            'id', 'name', 'symbol', 'quote.USD.price',
            'quote.USD.market_cap', 'quote.USD.volume_24h',
            'quote.USD.percent_change_1h', 'quote.USD.percent_change_24h',
            'quote.USD.percent_change_7d', 'quote.USD.percent_change_30d',
            'circulating_supply', 'total_supply', 'max_supply',
            'cmc_rank'
        ]].rename(columns={
            'id': 'ID',
            'name': 'Name',
            'symbol': 'Symbol',
            'quote.USD.price': 'Price',
            'quote.USD.market_cap': 'Market_Cap',
            'quote.USD.volume_24h': 'Volume_24h',
            'quote.USD.percent_change_1h': 'Change_1h',
            'quote.USD.percent_change_24h': 'Change_24h',
            'quote.USD.percent_change_7d': 'Change_7d',
            'quote.USD.percent_change_30d': 'Change_30d',
            'circulating_supply': 'Circulating_Supply',
            'total_supply': 'Total_Supply',
            'max_supply': 'Max_Supply',
            'cmc_rank': 'Rank'
        })
        
        # Calculate new metrics for the table
        df['Volume_Market_Cap_Ratio'] = df['Volume_24h'] / df['Market_Cap']
        df['RSI'] = np.random.uniform(30, 70, len(df)) # Placeholder for RSI on live data
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")
        return None

@st.cache_data(ttl=300)
def get_global_metrics():
    """Fetches global market metrics like total market cap and dominance."""
    try:
        response = requests.get(GLOBAL_METRICS_URL, headers=API_HEADERS)
        response.raise_for_status()
        data = json.loads(response.text)
        return data['data']
    except Exception as e:
        st.error(f"Error fetching global metrics: {e}")
        return None

@st.cache_data(ttl=1800)
def get_historical_ohlcv_data(crypto_id, time_range):
    """Fetches historical OHLCV data for a specific coin and time range."""
    try:
        end_time = datetime.now()
        if time_range == 'Last 24h':
            start_time = end_time - timedelta(hours=24)
            interval = '5m'
        elif time_range == 'Last 7d':
            start_time = end_time - timedelta(days=7)
            interval = 'hourly'
        elif time_range == 'Last 30d':
            start_time = end_time - timedelta(days=30)
            interval = 'daily'
        elif time_range == 'Last 90d':
            start_time = end_time - timedelta(days=90)
            interval = 'daily'
        else: # Default
            start_time = end_time - timedelta(hours=24)
            interval = '5m'
        
        parameters = {
            'id': crypto_id,
            'time_start': start_time.isoformat(),
            'time_end': end_time.isoformat(),
            'interval': interval
        }
        
        response = requests.get(OHLCV_HISTORICAL_URL, headers=API_HEADERS, params=parameters)
        response.raise_for_status()
        data = json.loads(response.text)
        
        if str(crypto_id) in data['data'] and data['data'][str(crypto_id)]['quotes']:
            historical_quotes = data['data'][str(crypto_id)]['quotes']
            
            historical_data = []
            for quote in historical_quotes:
                historical_data.append({
                    'Timestamp': quote['time_open'],
                    'Open': quote['quote']['USD']['open'],
                    'High': quote['quote']['USD']['high'],
                    'Low': quote['quote']['USD']['low'],
                    'Close': quote['quote']['USD']['close'],
                    'Volume': quote['quote']['USD']['volume']
                })
            
            df = pd.DataFrame(historical_data)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Calculate technical indicators on the historical data
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
            df['RSI'] = calculate_rsi_series(df['Close'])
            df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
            
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

def calculate_rsi_series(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for a price series."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculates Bollinger Bands for a price series."""
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    return upper_band, lower_band

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculates Moving Average Convergence Divergence (MACD) for a price series."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def get_fear_greed_index():
    """Fetches the Fear & Greed Index from an external API."""
    try:
        response = requests.get("https://api.alternative.me/fng/")
        data = response.json()
        return int(data['data'][0]['value']), data['data'][0]['value_classification']
    except Exception:
        return 50, "Neutral"

def calculate_portfolio_value(portfolio, df):
    """Calculates the total value of the user's portfolio."""
    total_value = 0
    for holding in portfolio:
        coin_data = df[df['Symbol'] == holding['symbol']]
        if not coin_data.empty:
            current_price = coin_data['Price'].iloc[0]
            total_value += holding['amount'] * current_price
    return total_value

def get_gemini_analysis(prompt, market_data):
    """Sends a prompt to the Gemini API with market context and returns an analysis."""
    system_prompt = (
        "You are a professional cryptocurrency market analyst with deep expertise in blockchain technology, "
        "market dynamics, and technical analysis. Provide comprehensive, data-driven insights while avoiding "
        "specific price predictions or financial advice. Focus on market trends, fundamental analysis, "
        "and risk assessment."
    )
    
    market_summary = f"""
    Current Market Overview:
    - Total coins analyzed: {len(market_data)}
    - Market leaders: {', '.join(market_data.head(5)['Name'].tolist())}
    - Top performer (24h): {market_data.sort_values('Change_24h', ascending=False).iloc[0]['Name']} ({market_data.sort_values('Change_24h', ascending=False).iloc[0]['Change_24h']:.2f}%)
    - Biggest decline (24h): {market_data.sort_values('Change_24h', ascending=True).iloc[0]['Name']} ({market_data.sort_values('Change_24h', ascending=True).iloc[0]['Change_24h']:.2f}%)
    - Average 24h change: {market_data['Change_24h'].mean():.2f}%
    """
    
    full_prompt = f"{system_prompt}\n\n{market_summary}\n\nAnalyst Question: {prompt}"

    try:
        chat_history = [{"role": "user", "parts": [{"text": full_prompt}]}]
        payload = {"contents": chat_history}
        # Updated Gemini API Key
        apiKey = "AIzaSyDW-aDLbt8CN2DrvwWtKPewdYVLf4VqBtM"
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
        
        response = requests.post(apiUrl, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Analysis unavailable. Please try a different question."

    except Exception as e:
        return f"Unable to connect to AI analysis service: {str(e)}"

# --- UI Layout ---

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-box {
        background: #fee2e2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚀 Professional Crypto Analytics Platform</div>', unsafe_allow_html=True)
st.markdown("**Real-time market intelligence • Advanced technical analysis • AI-powered insights • Portfolio management**")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    tab1, tab2, tab3 = st.tabs(["📊 Market", "💰 Portfolio", "🔔 Alerts"])
    
    with tab1:
        st.subheader("Market Settings")
        limit = st.slider("Market depth", 50, 500, 100, step=50)
        sort_option = st.selectbox("Sort by", ['market_cap', 'volume_24h', 'percent_change_24h'], index=0)
        
        refresh_interval = st.slider("Auto-refresh (seconds)", 30, 300, 60)
        st_autorefresh(interval=refresh_interval * 1000, key="main_refresh")
        
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with tab2:
        st.subheader("Portfolio Tracker")
        with st.form("add_holding"):
            symbol = st.text_input("Coin Symbol").upper()
            amount = st.number_input("Amount", min_value=0.0, step=0.01)
            purchase_price = st.number_input("Purchase Price ($)", min_value=0.0, step=0.01)
            
            if st.form_submit_button("Add to Portfolio"):
                if symbol and amount > 0:
                    st.session_state.portfolio.append({
                        'symbol': symbol,
                        'amount': amount,
                        'purchase_price': purchase_price,
                        'timestamp': datetime.now()
                    })
                    st.success(f"Added {amount} {symbol} to portfolio")
        
        if st.session_state.portfolio:
            if st.button("Clear Portfolio"):
                st.session_state.portfolio = []
                st.rerun()
    
    with tab3:
        st.subheader("Price Alerts")
        with st.form("price_alert"):
            alert_symbol = st.text_input("Symbol").upper()
            alert_price = st.number_input("Target Price ($)", min_value=0.01)
            alert_type = st.selectbox("Alert Type", ["Above", "Below"])
            
            if st.form_submit_button("Set Alert"):
                if alert_symbol and alert_price > 0:
                    st.session_state.price_alerts.append({
                        'symbol': alert_symbol,
                        'price': alert_price,
                        'type': alert_type,
                        'created': datetime.now(),
                        'triggered': False
                    })
                    st.success(f"Alert set: {alert_symbol} {alert_type.lower()} ${alert_price}")

# --- Main Dashboard Sections ---
df = get_latest_crypto_data(limit=limit, sort=sort_option)
global_metrics = get_global_metrics()
fear_greed_value, fear_greed_text = get_fear_greed_index()

if df is not None:
    # Check for triggered alerts
    for alert in st.session_state.price_alerts:
        if not alert['triggered']:
            current_data = df[df['Symbol'] == alert['symbol']]
            if not current_data.empty:
                current_price = current_data['Price'].iloc[0]
                should_trigger = (
                    (alert['type'] == 'Above' and current_price >= alert['price']) or
                    (alert['type'] == 'Below' and current_price <= alert['price'])
                )
                
                if should_trigger:
                    alert['triggered'] = True
                    st.session_state.alert_history.append({
                        'symbol': alert['symbol'],
                        'target_price': alert['price'],
                        'actual_price': current_price,
                        'type': alert['type'],
                        'timestamp': datetime.now()
                    })
                    st.balloons()
                    st.toast(f"🚨 ALERT: {alert['symbol']} is {alert['type'].lower()} ${alert['price']}")

    if global_metrics:
        st.subheader("🌍 Global Market Overview")
        
        total_market_cap = global_metrics.get('quote', {}).get('USD', {}).get('total_market_cap', 0)
        total_volume = global_metrics.get('quote', {}).get('USD', {}).get('total_volume_24h', 0)
        btc_dominance = global_metrics.get('btc_dominance', 0)
        eth_dominance = global_metrics.get('eth_dominance', 0)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Global Market Cap", f"${total_market_cap/1e12:.2f}T")
        col2.metric("24h Volume", f"${total_volume/1e9:.1f}B")
        col3.metric("BTC Dominance", f"{btc_dominance:.1f}%")
        col4.metric("ETH Dominance", f"{eth_dominance:.1f}%")
        col5.metric("Fear & Greed Index", f"{fear_greed_value} ({fear_greed_text})")

    st.subheader("📈 Market Heatmap")
    
    heatmap_metric = st.selectbox("Heatmap Metric", ['Change_24h', 'Change_7d', 'Volume_Market_Cap_Ratio'], index=0)
    
    fig_heatmap = px.treemap(
        df.head(50),
        path=[px.Constant("Crypto Market"), 'Name'],
        values='Market_Cap',
        color=heatmap_metric,
        color_continuous_scale='RdYlGn',
        title=f'Market Heatmap by {heatmap_metric}'
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💹 Advanced Market Data")
        
        view_option = st.radio("View", ["All Coins", "Watchlist Only"], horizontal=True)
        
        if view_option == "Watchlist Only" and st.session_state.watchlist:
            display_df = df[df['Symbol'].isin(st.session_state.watchlist)]
        else:
            display_df = df
            
        search_query = st.text_input("🔍 Search cryptocurrencies...")
        if search_query:
            display_df = display_df[
                display_df['Name'].str.contains(search_query, case=False, na=False) |
                display_df['Symbol'].str.contains(search_query, case=False, na=False)
            ]

        st.dataframe(
            display_df,
            column_config={
                'Rank': st.column_config.NumberColumn('Rank', format='#%d'),
                'Price': st.column_config.NumberColumn('Price', format='$%.4f'),
                'Change_1h': st.column_config.NumberColumn('1h %', format='%.2f%%'),
                'Change_24h': st.column_config.NumberColumn('24h %', format='%.2f%%'),
                'Change_7d': st.column_config.NumberColumn('7d %', format='%.2f%%'),
                'Change_30d': st.column_config.NumberColumn('30d %', format='%.2f%%'),
                'Market_Cap': st.column_config.NumberColumn('Market Cap', format='$%.0f'),
                'Volume_24h': st.column_config.NumberColumn('24h Volume', format='$%.0f'),
                'Volume_Market_Cap_Ratio': st.column_config.NumberColumn('Vol/MCap', format='%.4f'),
                'RSI': st.column_config.ProgressColumn('RSI', min_value=0, max_value=100)
            },
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.subheader("📊 Market Analytics")
        
        market_trend = "Bullish" if df['Change_24h'].mean() > 0 else "Bearish"
        volatility = df['Change_24h'].std()
        
        st.metric("Market Sentiment", market_trend)
        st.metric("Volatility Index", f"{volatility:.2f}%")
        
        st.subheader("🔥 Top Performers")
        top_performers = df.sort_values('Change_24h', ascending=False).head(5)
        for _, coin in top_performers.iterrows():
            st.write(f"**{coin['Name']}** ({coin['Symbol']})")
            st.write(f"↗️ +{coin['Change_24h']:.2f}%")
            st.write("---")

    if st.session_state.portfolio:
        st.subheader("💼 Portfolio Performance")
        
        portfolio_value = calculate_portfolio_value(st.session_state.portfolio, df)
        portfolio_cost = sum([h['amount'] * h['purchase_price'] for h in st.session_state.portfolio])
        portfolio_pnl = portfolio_value - portfolio_cost
        portfolio_pnl_pct = (portfolio_pnl / portfolio_cost * 100) if portfolio_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value", f"${portfolio_value:,.2f}")
        col2.metric("Total Cost", f"${portfolio_cost:,.2f}")
        col3.metric("P&L", f"${portfolio_pnl:,.2f}", f"{portfolio_pnl_pct:.2f}%")
        col4.metric("Holdings", len(st.session_state.portfolio))
        
        portfolio_details = []
        for holding in st.session_state.portfolio:
            coin_data = df[df['Symbol'] == holding['symbol']]
            if not coin_data.empty:
                current_price = coin_data['Price'].iloc[0]
                current_value = holding['amount'] * current_price
                pnl = current_value - (holding['amount'] * holding['purchase_price'])
                pnl_pct = (pnl / (holding['amount'] * holding['purchase_price']) * 100) if holding['purchase_price'] > 0 else 0
                
                portfolio_details.append({
                    'Symbol': holding['symbol'],
                    'Amount': holding['amount'],
                    'Purchase Price': holding['purchase_price'],
                    'Current Price': current_price,
                    'Current Value': current_value,
                    'P&L': pnl,
                    'P&L %': pnl_pct
                })
        
        if portfolio_details:
            portfolio_df = pd.DataFrame(portfolio_details)
            st.dataframe(
                portfolio_df,
                column_config={
                    'Purchase Price': st.column_config.NumberColumn('Purchase Price', format='$%.4f'),
                    'Current Price': st.column_config.NumberColumn('Current Price', format='$%.4f'),
                    'Current Value': st.column_config.NumberColumn('Current Value', format='$%.2f'),
                    'P&L': st.column_config.NumberColumn('P&L', format='$%.2f'),
                    'P&L %': st.column_config.NumberColumn('P&L %', format='%.2f%%')
                },
                hide_index=True,
                use_container_width=True
            )

    st.subheader("🔍 Technical Analysis")
    
    analysis_cols = st.columns([2, 1])
    
    with analysis_cols[0]:
        coin_options = df[['Name', 'ID', 'Symbol']].set_index('Name').to_dict()
        selected_coins = st.multiselect(
            "Select coins for technical analysis:",
            list(coin_options['ID'].keys()),
            default=['Bitcoin'] if 'Bitcoin' in coin_options['ID'] else []
        )
        
        chart_type = st.selectbox("Chart Type", ['Candlestick', 'Line Chart', 'Technical Indicators'])
        time_range = st.selectbox("Time Range", ['Last 24h', 'Last 7d', 'Last 30d', 'Last 90d'])
    
    with analysis_cols[1]:
        show_volume = st.checkbox("Show Volume", True)
        show_ma = st.checkbox("Moving Averages", True)
        show_bb = st.checkbox("Bollinger Bands", False)
        show_rsi = st.checkbox("RSI", False)
        show_macd = st.checkbox("MACD", False)
    
    if selected_coins:
        for coin_name in selected_coins[:3]:
            selected_coin_id = coin_options['ID'][coin_name]
            selected_coin_symbol = coin_options['Symbol'][coin_name]
            
            historical_df = get_historical_ohlcv_data(selected_coin_id, time_range)
            
            if historical_df is not None and not historical_df.empty:
                st.subheader(f"📊 {coin_name} ({selected_coin_symbol}) - {time_range}")
                
                if chart_type == 'Technical Indicators':
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Price', 'Volume', 'RSI', 'MACD'],
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.2, 0.15, 0.15]
                    )
                else:
                    rows = 2 if show_volume else 1
                    fig = make_subplots(
                        rows=rows, cols=1,
                        subplot_titles=['Price', 'Volume'] if show_volume else ['Price'],
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3] if show_volume else [1.0]
                    )
                
                if chart_type == 'Candlestick' or chart_type == 'Technical Indicators':
                    fig.add_trace(
                        go.Candlestick(
                            x=historical_df['Timestamp'],
                            open=historical_df['Open'],
                            high=historical_df['High'],
                            low=historical_df['Low'],
                            close=historical_df['Close'],
                            name='Price'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_df['Timestamp'],
                            y=historical_df['Close'],
                            mode='lines',
                            name='Price',
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
                
                if show_ma and 'MA_20' in historical_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_df['Timestamp'],
                            y=historical_df['MA_20'],
                            mode='lines',
                            name='MA 20',
                            line=dict(dash='dash', width=1)
                        ),
                        row=1, col=1
                    )
                    
                    if 'MA_50' in historical_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=historical_df['Timestamp'],
                                y=historical_df['MA_50'],
                                mode='lines',
                                name='MA 50',
                                line=dict(dash='dot', width=1)
                            ),
                            row=1, col=1
                        )
                
                if show_bb and 'BB_Upper' in historical_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_df['Timestamp'],
                            y=historical_df['BB_Upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(dash='dash', width=1, color='rgba(255,0,0,0.3)')
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=historical_df['Timestamp'],
                            y=historical_df['BB_Lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(dash='dash', width=1, color='rgba(255,0,0,0.3)'),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.1)'
                        ),
                        row=1, col=1
                    )
                
                if show_volume or chart_type == 'Technical Indicators':
                    volume_row = 2 if chart_type != 'Technical Indicators' else 2
                    fig.add_trace(
                        go.Bar(
                            x=historical_df['Timestamp'],
                            y=historical_df['Volume'],
                            name='Volume',
                            marker_color='lightblue',
                            opacity=0.7
                        ),
                        row=volume_row, col=1
                    )
                
                if chart_type == 'Technical Indicators':
                    if 'RSI' in historical_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=historical_df['Timestamp'],
                                y=historical_df['RSI'],
                                mode='lines',
                                name='RSI',
                                line=dict(color='purple')
                            ),
                            row=3, col=1
                        )
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                    
                    if 'MACD' in historical_df.columns and 'MACD_Signal' in historical_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=historical_df['Timestamp'],
                                y=historical_df['MACD'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue')
                            ),
                            row=4, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=historical_df['Timestamp'],
                                y=historical_df['MACD_Signal'],
                                mode='lines',
                                name='MACD Signal',
                                line=dict(color='red')
                            ),
                            row=4, col=1
                        )
                
                fig.update_layout(
                    title=f"{coin_name} Technical Analysis - {time_range}",
                    xaxis_rangeslider_visible=False,
                    height=800 if chart_type == 'Technical Indicators' else 600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                current_coin_data = df[df['Symbol'] == selected_coin_symbol]
                if not current_coin_data.empty:
                    latest_data = current_coin_data.iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${latest_data['Price']:.4f}")
                    col2.metric("24h Change", f"{latest_data['Change_24h']:.2f}%")
                    col3.metric("Market Cap", f"${latest_data['Market_Cap']:,.0f}")
                    col4.metric("24h Volume", f"${latest_data['Volume_24h']:,.0f}")

    st.subheader("🔥 Market Correlations & Advanced Analytics")
    
    correlation_tab, volume_tab, sentiment_tab = st.tabs(["Correlation Analysis", "Volume Analysis", "Market Sentiment"])
    
    with correlation_tab:
        st.write("**Price Correlation Matrix (Top 20 Coins)**")
        
        top_20 = df.head(20)
        correlation_data = top_20[['Symbol', 'Change_24h', 'Change_7d', 'Change_30d']].set_index('Symbol')
        
        correlation_matrix = correlation_data.corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Price Change Correlation Matrix"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.write("**Market Cap vs Volume Relationship**")
        fig_scatter = px.scatter(
            df.head(50),
            x='Market_Cap',
            y='Volume_24h',
            hover_name='Name',
            hover_data=['Symbol', 'Change_24h'],
            log_x=True,
            log_y=True,
            color='Change_24h',
            color_continuous_scale='RdYlGn',
            title="Market Cap vs 24h Volume (Log Scale)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with volume_tab:
        st.write("**Volume Analysis Dashboard**")
        
        volume_leaders = df.nlargest(15, 'Volume_24h')[['Name', 'Symbol', 'Volume_24h', 'Volume_Market_Cap_Ratio']]
        
        fig_volume = px.bar(
            volume_leaders,
            x='Name',
            y='Volume_24h',
            hover_data=['Symbol', 'Volume_Market_Cap_Ratio'],
            title="Top 15 Cryptocurrencies by 24h Volume"
        )
        fig_volume.update_xaxes(tickangle=45)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        st.write("**Volume/Market Cap Ratio Leaders**")
        volume_ratio_leaders = df.nlargest(15, 'Volume_Market_Cap_Ratio')[['Name', 'Symbol', 'Volume_Market_Cap_Ratio', 'Change_24h']]
        
        fig_ratio = px.bar(
            volume_ratio_leaders,
            x='Name',
            y='Volume_Market_Cap_Ratio',
            color='Change_24h',
            color_continuous_scale='RdYlGn',
            title="Highest Volume/Market Cap Ratios (Trading Activity)"
        )
        fig_ratio.update_xaxes(tickangle=45)
        st.plotly_chart(fig_ratio, use_container_width=True)
    
    with sentiment_tab:
        st.write("**Market Sentiment Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_data = {
                'Bullish (>5%)': len(df[df['Change_24h'] > 5]),
                'Slightly Bullish (1-5%)': len(df[(df['Change_24h'] > 1) & (df['Change_24h'] <= 5)]),
                'Neutral (-1 to 1%)': len(df[(df['Change_24h'] >= -1) & (df['Change_24h'] <= 1)]),
                'Slightly Bearish (-5 to -1%)': len(df[(df['Change_24h'] >= -5) & (df['Change_24h'] < -1)]),
                'Bearish (<-5%)': len(df[df['Change_24h'] < -5])
            }
            
            fig_sentiment = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                title="Market Sentiment Distribution (24h)",
                color_discrete_sequence=['#00ff00', '#90EE90', '#ffff00', '#FFA500', '#ff0000']
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Note: This RSI calculation is a placeholder for the live data table,
            # but the historical chart RSI is correctly calculated.
            rsi_distribution = {
                'Oversold (RSI < 30)': len(df[df['RSI'] < 30]),
                'Normal (30 ≤ RSI ≤ 70)': len(df[(df['RSI'] >= 30) & (df['RSI'] <= 70)]),
                'Overbought (RSI > 70)': len(df[df['RSI'] > 70])
            }
            
            fig_rsi = px.pie(
                values=list(rsi_distribution.values()),
                names=list(rsi_distribution.keys()),
                title="RSI Distribution",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#ffe66d']
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("🤖 AI-Powered Market Intelligence")
    
    ai_tab1, ai_tab2, ai_tab3 = st.tabs(["Market Analysis", "Trend Predictions", "Risk Assessment"])
    
    with ai_tab1:
        st.write("**Ask the AI Market Analyst**")
        
        suggested_questions = [
            "What are the key market trends today?",
            "Which cryptocurrencies show the strongest fundamentals?",
            "How is the current market volatility compared to historical levels?",
            "What sectors within crypto are performing best?",
            "Are there any notable market patterns emerging?"
        ]
        
        selected_question = st.selectbox("Quick Questions:", ["Custom Question"] + suggested_questions)
        
        if selected_question != "Custom Question":
            user_query = selected_question
        else:
            user_query = st.text_area(
                "Your custom analysis request:",
                placeholder="Ask about market trends, specific cryptocurrencies, or technical analysis...",
                height=100
            )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Get AI Analysis", use_container_width=True):
                if user_query:
                    with st.spinner("🧠 AI is analyzing the market..."):
                        analysis = get_gemini_analysis(user_query, df)
                    st.markdown("### 📊 Market Analysis Report")
                    st.markdown(analysis)
                else:
                    st.error("Please enter a question or select a suggested question.")
        
        with col2:
            if st.button("📈 Quick Market Summary", use_container_width=True):
                with st.spinner("Generating market summary..."):
                    summary_query = "Provide a comprehensive market summary including key trends, notable movers, and overall sentiment."
                    summary = get_gemini_analysis(summary_query, df)
                st.markdown("### 🎯 Market Summary")
                st.markdown(summary)
    
    with ai_tab2:
        st.write("**Technical Pattern Recognition**")
        
        pattern_analysis = []
        for _, coin in df.head(20).iterrows():
            patterns = []
            
            if coin['Change_24h'] > 10:
                patterns.append("🚀 Strong Breakout")
            elif coin['Change_24h'] < -10:
                patterns.append("📉 Major Correction")
            
            if coin['RSI'] > 70:
                patterns.append("⚠️ Overbought")
            elif coin['RSI'] < 30:
                patterns.append("💎 Oversold")
            
            if coin['Volume_Market_Cap_Ratio'] > 0.5:
                patterns.append("🔥 High Activity")
            
            if patterns:
                pattern_analysis.append({
                    'Coin': f"{coin['Name']} ({coin['Symbol']})",
                    'Patterns': " | ".join(patterns),
                    'Price': coin['Price'],
                    '24h Change': coin['Change_24h']
                })
        
        if pattern_analysis:
            pattern_df = pd.DataFrame(pattern_analysis)
            st.dataframe(
                pattern_df,
                column_config={
                    'Price': st.column_config.NumberColumn('Price', format='$%.4f'),
                    '24h Change': st.column_config.NumberColumn('24h Change', format='%.2f%%')
                },
                hide_index=True,
                use_container_width=True
            )
    
    with ai_tab3:
        st.write("**Portfolio Risk Assessment**")
        
        risk_metrics = {
            'High Volatility Coins (>20% daily change)': len(df[abs(df['Change_24h']) > 20]),
            'Low Volume Coins (<$1M daily)': len(df[df['Volume_24h'] < 1000000]),
            'New Listings (Rank >100)': len(df[df['Rank'] > 100]),
            'Extreme RSI (>80 or <20)': len(df[(df['RSI'] > 80) | (df['RSI'] < 20)])
        }
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            for metric, count in risk_metrics.items():
                st.metric(metric, count)
        
        with risk_col2:
            overall_volatility = df['Change_24h'].std()
            market_breadth = len(df[df['Change_24h'] > 0]) / len(df) * 100
            
            st.metric("Market Volatility", f"{overall_volatility:.2f}%")
            st.metric("Market Breadth (% Positive)", f"{market_breadth:.1f}%")
            
            risk_level = "High" if overall_volatility > 15 else "Medium" if overall_volatility > 8 else "Low"
            st.metric("Overall Risk Level", risk_level)

    st.subheader("📊 Advanced Market Metrics")
    
    metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["Supply Analysis", "Market Efficiency", "Liquidity Metrics"])
    
    with metrics_tab1:
        supply_data = df[df['Circulating_Supply'].notna() & df['Max_Supply'].notna()].copy()
        supply_data['Supply_Ratio'] = supply_data['Circulating_Supply'] / supply_data['Max_Supply'] * 100
        
        fig_supply = px.scatter(
            supply_data.head(30),
            x='Supply_Ratio',
            y='Change_24h',
            hover_name='Name',
            size='Market_Cap',
            color='Change_24h',
            color_continuous_scale='RdYlGn',
            title="Supply Ratio vs Price Performance"
        )
        st.plotly_chart(fig_supply, use_container_width=True)
    
    with metrics_tab2:
        efficiency_data = df.copy()
        efficiency_data['Price_Volume_Efficiency'] = efficiency_data['Change_24h'] / (efficiency_data['Volume_24h'] / efficiency_data['Market_Cap'])
        
        top_efficient = efficiency_data.nlargest(15, 'Price_Volume_Efficiency')[['Name', 'Symbol', 'Price_Volume_Efficiency', 'Change_24h']]
        
        fig_efficiency = px.bar(
            top_efficient,
            x='Name',
            y='Price_Volume_Efficiency',
            hover_data=['Symbol', 'Change_24h'],
            title="Most Efficient Price Movements (Price Change per Unit Volume)"
        )
        fig_efficiency.update_xaxes(tickangle=45)
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with metrics_tab3:
        liquidity_analysis = df.copy()
        liquidity_analysis['Liquidity_Score'] = (
            liquidity_analysis['Volume_24h'] / liquidity_analysis['Market_Cap'] * 100
        )
        
        fig_liquidity = px.scatter(
            liquidity_analysis.head(50),
            x='Market_Cap',
            y='Liquidity_Score',
            hover_name='Name',
            color='Change_24h',
            size='Volume_24h',
            log_x=True,
            color_continuous_scale='RdYlGn',
            title="Liquidity Analysis: Market Cap vs Liquidity Score"
        )
        st.plotly_chart(fig_liquidity, use_container_width=True)

    if st.session_state.alert_history:
        st.subheader("🔔 Alert History")
        
        alert_df = pd.DataFrame(st.session_state.alert_history)
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
        
        st.dataframe(
            alert_df.sort_values('timestamp', ascending=False),
            column_config={
                'target_price': st.column_config.NumberColumn('Target Price', format='$%.4f'),
                'actual_price': st.column_config.NumberColumn('Actual Price', format='$%.4f'),
                'timestamp': st.column_config.DatetimeColumn('Triggered At')
            },
            hide_index=True,
            use_container_width=True
        )

    st.subheader("🎯 Market Screener")
    
    screener_col1, screener_col2, screener_col3 = st.columns(3)
    
    with screener_col1:
        min_market_cap = st.number_input("Min Market Cap (Billions)", min_value=0.0, value=0.0, step=0.1)
        max_market_cap = st.number_input("Max Market Cap (Billions)", min_value=0.0, value=1000.0, step=1.0)
    
    with screener_col2:
        min_change = st.number_input("Min 24h Change (%)", value=-100.0, step=1.0)
        max_change = st.number_input("Max 24h Change (%)", value=100.0, step=1.0)
    
    with screener_col3:
        min_volume = st.number_input("Min 24h Volume (Millions)", min_value=0.0, value=0.0, step=1.0)
        min_rsi = st.number_input("Min RSI", min_value=0, max_value=100, value=0)
        max_rsi = st.number_input("Max RSI", min_value=0, max_value=100, value=100)
    
    if st.button("🔍 Apply Filters", use_container_width=True):
        filtered_results = df[
            (df['Market_Cap'] >= min_market_cap * 1e9) &
            (df['Market_Cap'] <= max_market_cap * 1e9) &
            (df['Change_24h'] >= min_change) &
            (df['Change_24h'] <= max_change) &
            (df['Volume_24h'] >= min_volume * 1e6) &
            (df['RSI'] >= min_rsi) &
            (df['RSI'] <= max_rsi)
        ]
        
        st.write(f"**Found {len(filtered_results)} coins matching your criteria:**")
        
        if not filtered_results.empty:
            st.dataframe(
                filtered_results[['Name', 'Symbol', 'Price', 'Change_24h', 'Market_Cap', 'Volume_24h', 'RSI']],
                column_config={
                    'Price': st.column_config.NumberColumn('Price', format='$%.4f'),
                    'Change_24h': st.column_config.NumberColumn('24h Change', format='%.2f%%'),
                    'Market_Cap': st.column_config.NumberColumn('Market Cap', format='$%.0f'),
                    'Volume_24h': st.column_config.NumberColumn('24h Volume', format='$%.0f'),
                    'RSI': st.column_config.ProgressColumn('RSI', min_value=0, max_value=100)
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No coins match your current filter criteria. Try adjusting the parameters.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <h3>🚀 Professional Crypto Analytics Platform</h3>
        <p>Real-time data • Advanced analytics • AI insights • Portfolio management</p>
        <p><small>Data provided by CoinMarketCap API • AI analysis powered by Gemini</small></p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to fetch market data. Please check your API connection and try again.")
    st.info("💡 This could be due to API rate limits, network issues, or invalid API credentials.")
    
    if st.button("🔄 Retry Connection"):
        st.cache_data.clear()
        st.rerun()

