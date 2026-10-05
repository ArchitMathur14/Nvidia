import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import pandas_datareader as pdr
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SAPM-1 Analysis: NVIDIA",
    page_icon="📊",
    layout="wide"
)

# --- DATA CACHING TO PREVENT SLOW LOAD TIMES ---
@st.cache_data
def load_macro_data():
    """Fetch real US GDP and Federal Funds Rate from FRED."""
    start = datetime.datetime(2018, 1, 1)
    end = datetime.datetime.now()
    try:
        gdp = pdr.DataReader('GDP', 'fred', start, end)
        rates = pdr.DataReader('FEDFUNDS', 'fred', start, end)
        return gdp, rates
    except Exception:
        return None, None

@st.cache_data
def load_stock_data():
    """Fetch real historical stock data."""
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=5*365) # 5 years
    
    try:
        nvda = yf.Ticker('NVDA').history(start=start, end=end)
        sp500 = yf.Ticker('^GSPC').history(start=start, end=end)['Close']
        smh = yf.Ticker('SMH').history(start=start, end=end)['Close']
        
        # Main comparison dataframe
        df_comp = pd.DataFrame({'NVDA': nvda['Close'], 'S&P500': sp500, 'SMH': smh})
        df_comp.index = df_comp.index.tz_localize(None)
        
        # NVDA specific dataframe for volume/candlestick
        nvda.index = nvda.index.tz_localize(None)
        
        return df_comp.dropna(), nvda
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# Load the data
gdp_df, rates_df = load_macro_data()
comp_df, nvda_df = load_stock_data()

# --- SIDEBAR: REPORT DETAILS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=150)
st.sidebar.title("Analysis Report")
st.sidebar.markdown("**Submitted by:** Archit Mathur")
st.sidebar.markdown("**Program:** PGDM 2025–27")
st.sidebar.markdown("**Institution:** FORE School of Management")
st.sidebar.markdown("**Submitted to:** Bhaskar Chhimwal (Prof.)")
st.sidebar.markdown("**Course:** SAPM-1")
st.sidebar.divider()

# Navigation
st.sidebar.subheader("Navigation")
section = st.sidebar.radio("Go to section:", [
    "1. The US Economy", 
    "2. The Semiconductor Industry", 
    "3. NVIDIA Corporation"
])

# --- SECTION 1: THE US ECONOMY ---
if section == "1. The US Economy":
    st.title("1. The US Economy")
    st.markdown("### Historical Macroeconomic Indicators")
    st.write("An overview of the macroeconomic environment impacting the technology and semiconductor sectors.")
    
    if gdp_df is not None and not gdp_df.empty:
        fig_gdp = px.line(gdp_df, x=gdp_df.index, y='GDP', title='US Gross Domestic Product (Billions of Dollars)')
        fig_gdp.update_layout(xaxis_title="Date", yaxis_title="GDP (Billions)")
        st.plotly_chart(fig_gdp, use_container_width=True)
        
        fig_rates = px.line(rates_df, x=rates_df.index, y='FEDFUNDS', title='Effective Federal Funds Rate (%)', color_discrete_sequence=['red'])
        fig_rates.update_layout(xaxis_title="Date", yaxis_title="Interest Rate (%)")
        st.plotly_chart(fig_rates, use_container_width=True)
    else:
        st.error("Failed to load FRED data. Please check your internet connection or API limits.")

# --- SECTION 2: THE SEMICONDUCTOR INDUSTRY ---
elif section == "2. The Semiconductor Industry":
    st.title("2. The Semiconductor Industry")
    st.markdown("### Industry vs. Broader Market Performance")
    st.write("Comparing the historical performance of the VanEck Semiconductor ETF (SMH) against the broader S&P 500 index.")
    
    if not comp_df.empty:
        # Normalize prices to 100 at the start of the period for easy comparison
        normalized_df = (comp_df / comp_df.iloc[0]) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['SMH'], mode='lines', name='Semiconductor ETF (SMH)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['S&P500'], mode='lines', name='S&P 500 Market', line=dict(color='gray')))
        
        fig.update_layout(title='5-Year Normalized Performance (Base 100)', yaxis_title='Normalized Price', hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
         st.error("Failed to fetch historical stock data. The API may be temporarily blocking requests.")

# --- SECTION 3: NVIDIA CORPORATION ---
elif section == "3. NVIDIA Corporation":
    st.title("3. NVIDIA Corporation")
    st.markdown("### Historical Price and Volume Analysis")
    
    if not nvda_df.empty:
        # Interactive time horizon selection
        time_horizon = st.selectbox("Select Time Horizon:", ["1 Year", "3 Years", "5 Years"], index=2)
        
        # Filter data based on selection
        cutoff_date = datetime.datetime.now()
        if time_horizon == "1 Year":
            cutoff_date -= datetime.timedelta(days=365)
        elif time_horizon == "3 Years":
            cutoff_date -= datetime.timedelta(days=3*365)
        else:
            cutoff_date -= datetime.timedelta(days=5*365)
            
        filtered_nvda = nvda_df[nvda_df.index >= cutoff_date]
        
        # Calculate Simple Moving Averages (SMA)
        filtered_nvda['SMA_50'] = filtered_nvda['Close'].rolling(window=50).mean()
        filtered_nvda['SMA_200'] = filtered_nvda['Close'].rolling(window=200).mean()

        # Plotly Candlestick Chart with Volume
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=filtered_nvda.index,
                open=filtered_nvda['Open'],
                high=filtered_nvda['High'],
                low=filtered_nvda['Low'],
                close=filtered_nvda['Close'],
                name='NVDA Price'))
        
        # Moving Averages
        fig.add_trace(go.Scatter(x=filtered_nvda.index, y=filtered_nvda['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=filtered_nvda.index, y=filtered_nvda['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='blue', width=1)))
        
        fig.update_layout(
            title=f"NVIDIA (NVDA) Stock Performance ({time_horizon})",
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume Chart
        st.markdown("### Trading Volume")
        fig_vol = px.bar(filtered_nvda, x=filtered_nvda.index, y='Volume', title='Daily Trading Volume')
        fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume", height=300)
        st.plotly_chart(fig_vol, use_container_width=True)
        
    else:
        st.error("Failed to fetch historical stock data. The API may be temporarily blocking requests.")
