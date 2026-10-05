import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import yfinance as yf
import pandas_datareader as pdr
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SAPM-1 Analysis: NVIDIA",
    page_icon="📈",
    layout="wide"
)

# --- DATA CACHING TO PREVENT SLOW LOAD TIMES ---
@st.cache_data
def load_macro_data():
    """Fetch real US GDP and Federal Funds Rate from FRED."""
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime.now()
    try:
        gdp = pdr.DataReader('GDP', 'fred', start, end)
        rates = pdr.DataReader('FEDFUNDS', 'fred', start, end)
        return gdp, rates
    except Exception:
        return None, None

@st.cache_data
def load_stock_data():
    """Fetch real stock data using a more stable yfinance method."""
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=5*365) # 5 years
    
    try:
        # Fetch history individually to avoid MultiIndex parsing errors
        nvda = yf.Ticker('NVDA').history(start=start, end=end)['Close']
        sp500 = yf.Ticker('^GSPC').history(start=start, end=end)['Close']
        smh = yf.Ticker('SMH').history(start=start, end=end)['Close']
        
        df = pd.DataFrame({'NVDA': nvda, 'S&P500': sp500, 'SMH': smh})
        # Remove timezone awareness to prevent alignment issues
        df.index = df.index.tz_localize(None)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

# Load the data
gdp_df, rates_df = load_macro_data()
stock_df = load_stock_data()

# Calculate Daily Returns for SAPM Analysis safely
if not stock_df.empty:
    returns_df = stock_df.pct_change().dropna()
else:
    returns_df = pd.DataFrame()

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
    "3. NVIDIA Corp (CAPM & Monte Carlo)"
])

# --- SECTION 1: THE US ECONOMY ---
if section == "1. The US Economy":
    st.title("1. The US Economy")
    st.markdown("### Macroeconomic Indicators (Live FRED Data)")
    st.write("In Security Analysis, understanding the macroeconomic environment is the first step of the Top-Down Approach. Here we look at actual US GDP growth and Interest Rates.")
    
    if gdp_df is not None and not gdp_df.empty:
        fig_gdp = px.line(gdp_df, x=gdp_df.index, y='GDP', title='US Gross Domestic Product (Billions of Dollars)')
        st.plotly_chart(fig_gdp, use_container_width=True)
        
        fig_rates = px.line(rates_df, x=rates_df.index, y='FEDFUNDS', title='Effective Federal Funds Rate (%)', color_discrete_sequence=['red'])
        st.plotly_chart(fig_rates, use_container_width=True)
    else:
        st.error("Failed to load FRED data. Please check your internet connection or FRED API limits.")

# --- SECTION 2: THE SEMICONDUCTOR INDUSTRY ---
elif section == "2. The Semiconductor Industry":
    st.title("2. The Semiconductor Industry")
    st.markdown("### Industry vs. Broader Market Performance")
    st.write("Comparing the VanEck Semiconductor ETF (SMH) against the S&P 500 to show industry outperformance.")
    
    if not stock_df.empty:
        # Normalize prices to 100 at the start of the period for easy comparison
        normalized_df = (stock_df / stock_df.iloc[0]) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['SMH'], mode='lines', name='Semiconductor ETF (SMH)'))
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['S&P500'], mode='lines', name='S&P 500 Market', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['NVDA'], mode='lines', name='NVIDIA (NVDA)', line=dict(color='green')))
        
        fig.update_layout(title='5-Year Normalized Performance (Base 100)', yaxis_title='Normalized Price', hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
         st.error("Failed to fetch historical stock data from Yahoo Finance. The API may be temporarily blocking requests from the cloud server.")

# --- SECTION 3: NVIDIA PREDICTION MODEL ---
elif section == "3. NVIDIA Corp (CAPM & Monte Carlo)":
    st.title("3. NVIDIA Corporation")
    
    tab1, tab2 = st.tabs(["Linear Regression (CAPM Beta)", "Monte Carlo Price Simulation"])
    
    # --- TAB 1: CAPM BETA REGRESSION ---
    with tab1:
        st.markdown("### CAPM: Systematic Risk (Beta) Calculation")
        
        if returns_df.empty:
            st.error("Failed to fetch historical stock data from Yahoo Finance. The API may be temporarily blocking requests from the cloud server.")
        else:
            st.write("Instead of regressing price against time, we regress NVIDIA's daily returns against the S&P 500's daily returns. The slope of this line is the stock's **Beta**, a measure of volatility relative to the market.")
            
            X = returns_df[['S&P500']].values
            y = returns_df['NVDA'].values
            
            # Linear Regression Model
            model = LinearRegression()
            model.fit(X, y)
            beta = model.coef_[0]
            alpha = model.intercept_
            
            # Plotly Scatter & Trendline
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Daily Returns', marker=dict(color='rgba(0, 150, 0, 0.3)', size=5)))
            fig_beta.add_trace(go.Scatter(x=X.flatten(), y=model.predict(X), mode='lines', name=f'Regression Line (Beta = {beta:.2f})', line=dict(color='red', width=3)))
            
            fig_beta.update_layout(title="NVIDIA vs S&P 500 Daily Returns", xaxis_title="S&P 500 Returns", yaxis_title="NVIDIA Returns")
            st.plotly_chart(fig_beta, use_container_width=True)
            
            col1, col2 = st.columns(2)
            col1.metric("Calculated Beta (Systematic Risk)", f"{beta:.2f}")
            col2.metric("Calculated Alpha (Idiosyncratic Return)", f"{alpha:.5f}")
            st.caption("A Beta > 1 indicates that NVIDIA is more volatile than the broader market.")

    # --- TAB 2: MONTE CARLO SIMULATION ---
    with tab2:
        st.markdown("### Monte Carlo Future Price Simulation")
        
        if returns_df.empty:
            st.error("Failed to fetch historical stock data from Yahoo Finance. The API may be temporarily blocking requests from the cloud server.")
        else:
            st.write("Simulating future price paths based on historical Geometric Brownian Motion (drift and volatility).")
            
            days_to_simulate = st.slider("Forecasting Horizon (Days):", 10, 252, 60, step=10)
            num_simulations = st.selectbox("Number of Simulations:", [100, 500, 1000], index=1)
            
            # Simulation Parameters
            last_price = stock_df['NVDA'].iloc[-1]
            mu = returns_df['NVDA'].mean()
            sigma = returns_df['NVDA'].std()
            
            # Generate random paths
            np.random.seed(42)
            simulated_paths = np.zeros((days_to_simulate, num_simulations))
            simulated_paths[0] = last_price
            
            for t in range(1, days_to_simulate):
                rand_shocks = np.random.normal(loc=mu, scale=sigma, size=num_simulations)
                simulated_paths[t] = simulated_paths[t-1] * (1 + rand_shocks)
                
            # Plot paths
            fig_mc = go.Figure()
            # Plot a subset of paths to keep the browser fast (e.g., 50 paths)
            for i in range(min(50, num_simulations)):
                fig_mc.add_trace(go.Scatter(x=np.arange(days_to_simulate), y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(0, 150, 0, 0.1)'), showlegend=False))
            
            # Calculate and plot the average path
            avg_path = simulated_paths.mean(axis=1)
            fig_mc.add_trace(go.Scatter(x=np.arange(days_to_simulate), y=avg_path, mode='lines', name='Expected Average Path', line=dict(color='red', width=3)))
            
            fig_mc.update_layout(title=f"Monte Carlo Simulation ({num_simulations} iterations) for next {days_to_simulate} days", 
                                 xaxis_title="Days in Future", yaxis_title="Simulated Price (USD)")
            st.plotly_chart(fig_mc, use_container_width=True)
            
            st.info(f"Current Price: **${last_price:.2f}** | Expected Average Price in {days_to_simulate} days: **${avg_path[-1]:.2f}**")
