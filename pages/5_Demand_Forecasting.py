import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Demand Forecasting - NeuralRetail AI", page_icon="📈", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-title {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Demand Forecasting")
st.markdown("Group 3 Demand Forecasting & Inventory Dashboard. Predict future daily sales revenue.")

BASE_DIR = Path(__file__).resolve().parents[1]
forecast_csv_path = BASE_DIR / "outputs" / "sales_forecast.csv"
model_path = BASE_DIR / "models" / "best_forecast_model.pkl"

if not forecast_csv_path.exists() or not model_path.exists():
    st.warning("⚠️ Demand forecasting outputs or models not found. Please run the Demand Forecasting pipeline (`src/forecasting.py`) to generate this data.")
else:
    @st.cache_data
    def load_forecast_data(csv_path):
        df = pd.read_csv(csv_path)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
        
    @st.cache_resource
    def load_forecast_model(mod_path):
        return joblib.load(mod_path)
        
    df_fore = load_forecast_data(forecast_csv_path)
    model_dict = load_forecast_model(model_path)
    
    # 1. Model metrics display
    st.markdown("### 🏆 Best Performing Forecasting Model")
    
    best_model_name = model_dict["model_name"]
    best_metrics = model_dict["metrics"][best_model_name]
    
    col_info, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])
    with col_info:
        st.markdown(
            f"""
            <div style="background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3); border-radius:12px; padding:1.25rem; height: 100%;">
                <h4 style="margin-top:0; color:#818CF8;">Model Selected: {best_model_name}</h4>
                <p style="margin:0; color:#E2E8F0;">Selected automatically based on the lowest root mean squared error (RMSE) on the 30-day temporal test split.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Test RMSE</div><div class="metric-value">${best_metrics["RMSE"]:.2f}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Test MAE</div><div class="metric-value">${best_metrics["MAE"]:.2f}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Test MAPE</div><div class="metric-value">{best_metrics["MAPE"]:.2f}%</div></div>', unsafe_allow_html=True)
        
    # 2. Forecasting Visualizations
    st.markdown("### 📊 Daily Sales & 30-Day Forecast")
    
    # Filter dataset for plotting
    hist_df = df_fore[df_fore["Type"] == "Historical"]
    fore_df = df_fore[df_fore["Type"] == "Forecast"]
    
    fig = go.Figure()
    
    # Historical Actuals
    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["y"],
        mode="lines",
        name="Historical Sales",
        line=dict(color="#6366F1", width=1.5)
    ))
    
    # Historical Predicted (Test Split fits)
    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["Predicted"],
        mode="lines",
        name="Model Fit / Predictions",
        line=dict(color="#10B981", width=1.5, dash="dot")
    ))
    
    # Future Forecasted
    fig.add_trace(go.Scatter(
        x=fore_df["ds"], y=fore_df["Predicted"],
        mode="lines+markers",
        name="30-Day Future Forecast",
        line=dict(color="#F59E0B", width=2.5)
    ))
    
    # Shaded confidence intervals
    fig.add_trace(go.Scatter(
        x=fore_df["ds"], y=fore_df["yhat_upper"],
        mode="lines",
        line=dict(color="rgba(245,158,11,0)"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=fore_df["ds"], y=fore_df["yhat_lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.15)",
        line=dict(color="rgba(245,158,11,0)"),
        name="Confidence Interval (85%-115%)",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Revenue Forecast with Model Confidence Bands",
        xaxis_title="Timeline",
        yaxis_title="Total Daily Revenue ($)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Forecast values table
    st.markdown("### 📋 Predicted Future Sales Sheet")
    st.write("Below are the detailed daily revenue predictions for the next 30 days.")
    
    # Formatted display
    export_fore = fore_df[["ds", "Predicted", "yhat_lower", "yhat_upper"]].copy()
    export_fore.columns = ["Forecast Date", "Predicted Revenue", "Lower Bound", "Upper Bound"]
    export_fore["Day of Week"] = export_fore["Forecast Date"].dt.day_name()
    
    # Reorder columns
    export_fore = export_fore[["Forecast Date", "Day of Week", "Predicted Revenue", "Lower Bound", "Upper Bound"]]
    
    st.dataframe(
        export_fore.style.format({
            "Predicted Revenue": "${:,.2f}",
            "Lower Bound": "${:,.2f}",
            "Upper Bound": "${:,.2f}"
        }),
        use_container_width=True
    )
