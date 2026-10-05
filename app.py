import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SAPM-1 Analysis Report: NVIDIA",
    page_icon="📈",
    layout="wide"
)

# --- SIDEBAR: REPORT DETAILS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=150)
st.sidebar.title("Analysis Report")
st.sidebar.markdown("**Submitted by:** Archit Mathur")
st.sidebar.markdown("**Program:** PGDM 2025–27")
st.sidebar.markdown("**Institution:** FORE School of Management, New Delhi")
st.sidebar.markdown("**Submitted to:** Bhaskar Chhimwal (Prof.)")
st.sidebar.markdown("**Course:** SAPM-1")
st.sidebar.divider()

# Navigation
st.sidebar.subheader("Navigation")
section = st.sidebar.radio("Go to section:", [
    "1. The US Economy", 
    "2. The Semiconductor Industry", 
    "3. NVIDIA Corporation (Prediction Model)"
])

# --- SECTION 1: THE US ECONOMY ---
if section == "1. The US Economy":
    st.title("The US Economy")
    st.markdown("### Macroeconomic Indicators")
    st.write("This section analyzes the macroeconomic landscape impacting the tech sector.")
    
    # Simulated GDP Growth Data
    col1, col2, col3 = st.columns(3)
    col1.metric("GDP Growth Rate", "2.1%", "+0.2%")
    col2.metric("Inflation Rate (CPI)", "3.2%", "-0.1%")
    col3.metric("Interest Rate", "5.25%", "0.0%")
    
    st.subheader("Simulated Economic Trend")
    chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["GDP Forecast", "Consumer Spending"])
    st.line_chart(chart_data)

# --- SECTION 2: THE SEMICONDUCTOR INDUSTRY ---
elif section == "2. The Semiconductor Industry":
    st.title("The Semiconductor Industry")
    st.markdown("### Industry Overview & Market Share")
    st.write("An analysis of the global semiconductor supply chain and demand constraints.")
    
    # Simulated Market Share Donut Chart
    labels = ['NVIDIA', 'AMD', 'Intel', 'Others']
    values = [65, 15, 12, 8]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="Estimated GPU Data Center Market Share (%)")
    st.plotly_chart(fig, use_container_width=True)

# --- SECTION 3: NVIDIA PREDICTION MODEL ---
elif section == "3. NVIDIA Corporation (Prediction Model)":
    st.title("NVIDIA Corporation")
    st.markdown("### Interactive Stock Trend & Linear Regression Predictor")
    
    # Generate Mock Historical Data for NVIDIA
    np.random.seed(42)
    days = np.arange(1, 101)
    # Simulate an upward trend with volatility
    historical_price = 400 + 4.5 * days + np.random.normal(0, 25, 100)
    df = pd.DataFrame({"Day": days, "Price": historical_price})

    # Interactivity: Allow faculty to adjust the prediction window
    st.markdown("Use the slider below to predict NVIDIA's price trend for future days based on historical linear regression.")
    future_days = st.slider("Select forecasting horizon (Days):", min_value=5, max_value=60, value=30, step=5)

    # --- MACHINE LEARNING: LINEAR REGRESSION ---
    X = df[["Day"]]
    y = df["Price"]
    model = LinearRegression()
    model.fit(X, y)

    # Predict future values
    future_X = np.arange(101, 101 + future_days).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    # Predict historical trendline
    trendline_y = model.predict(X)

    # --- PLOTLY VISUALIZATION ---
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(x=df["Day"], y=df["Price"], 
                             mode='lines', name='Simulated Historical Price',
                             line=dict(color='gray', width=2)))
    
    # Historical Trendline
    fig.add_trace(go.Scatter(x=df["Day"], y=trendline_y, 
                             mode='lines', name='Historical Trendline',
                             line=dict(color='blue', width=2)))
    
    # Future Prediction
    fig.add_trace(go.Scatter(x=np.arange(101, 101 + future_days), y=future_y, 
                             mode='lines+markers', name='Future Prediction (Regression)',
                             line=dict(color='green', dash='dash', width=3)))

    fig.update_layout(
        title=f"NVIDIA Price Projection ({future_days}-Day Horizon)",
        xaxis_title="Trading Days",
        yaxis_title="Price (USD)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display Model Metrics
    st.subheader("Model Insights")
    st.write(f"**Calculated Daily Growth Coefficient:** ${model.coef_[0]:.2f} per day")
    st.write(f"**Projected Price on Day {100 + future_days}:** ${future_y[-1]:.2f}")
