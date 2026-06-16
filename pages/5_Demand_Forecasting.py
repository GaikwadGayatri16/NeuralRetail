import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Demand Forecasting - NeuralRetail AI", page_icon="📈", layout="wide")

# Custom premium CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .insight-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .insight-title {
        color: #818CF8;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Demand Forecasting")
st.markdown("Group 3 Demand Forecasting Dashboard. Forecast future daily transaction revenue volumes and evaluate model accuracy.")
st.divider()

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
    
    best_model_name = model_dict["model_name"]
    best_metrics = model_dict["metrics"][best_model_name]
    
    # Display top-level KPIs before charts
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Winning Forecast Model", best_model_name)
    col2.metric("Test RMSE (Error Margin)", f"${best_metrics['RMSE']:.2f}")
    col3.metric("Test MAE (Mean Abs Error)", f"${best_metrics['MAE']:.2f}")
    col4.metric("Test MAPE (Percentage Error)", f"{best_metrics['MAPE']:.2f}%")
    
    st.divider()
    
    # Filter dataset for plotting
    hist_df = df_fore[df_fore["Type"] == "Historical"]
    fore_df = df_fore[df_fore["Type"] == "Forecast"]
    
    # Create Tabs
    tab_hist, tab_forecast, tab_val, tab_future, tab_metrics = st.tabs([
        "📊 Historical Sales",
        "📈 Sales & 30-Day Forecast",
        "⚖️ Actual vs Predicted",
        "🔮 Future 30 Days Forecast",
        "📋 Forecast Metrics & Details"
    ])
    
    # 1. HISTORICAL SALES
    with tab_hist:
        st.subheader("Daily Sales History")
        fig_hist = px.line(
            hist_df, x="ds", y="y",
            title="Historical Daily Sales Revenue ($)",
            labels={"y": "Daily Revenue ($)", "ds": "Timeline"},
            color_discrete_sequence=["#6366F1"]
        )
        fig_hist.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Total Revenue ($)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Dynamic Insight
        avg_hist = hist_df["y"].mean()
        max_hist_row = hist_df.loc[hist_df["y"].idxmax()]
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💡 Historical Sales Insight</div>
                <p style="margin:0; color:#E2E8F0;">
                    Average historical daily sales are <b>${avg_hist:,.2f}</b>. 
                    The highest sales spike occurred on <b>{max_hist_row['ds'].strftime('%d %B %Y')}</b>, reaching <b>${max_hist_row['y']:,.2f}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 2. INTEGRATED FORECAST
    with tab_forecast:
        st.subheader("Integrated Daily Sales & 30-Day Projections")
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
        
        # Confidence interval bounds
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
            title="Complete Daily Revenue Forecast with Confidence Intervals",
            xaxis_title="Timeline",
            yaxis_title="Daily Revenue ($)",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💡 Forecast Model Fit Insight</div>
                <p style="margin:0; color:#E2E8F0;">
                    The dashboard shows the model fitting historical behavior closely (indicated by the dashed green line) 
                    and extending into a 30-day projection (orange line) with statistical variance bands based on seasonal variance.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 3. ACTUAL VS PREDICTED (VALIDATION SPLIT)
    with tab_val:
        st.subheader("Model Validation Accuracy Fit")
        
        # Remove NaN rows (e.g. initial lag features) for comparison
        clean_hist_df = hist_df.dropna(subset=["y", "Predicted"])
        
        fig_scatter = px.scatter(
            clean_hist_df, x="y", y="Predicted",
            trendline="ols",
            title="Comparison of Actual vs Predicted Daily Sales ($)",
            labels={"y": "Actual Sales ($)", "Predicted": "Predicted Sales ($)"},
            color_discrete_sequence=["#10B981"]
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Calculate Correlation
        corr = clean_hist_df["y"].corr(clean_hist_df["Predicted"])
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💡 Fit Accuracy Insight</div>
                <p style="margin:0; color:#E2E8F0;">
                    The correlation coefficient between predictions and actual targets is <b>{corr:.4f}</b>. 
                    The diagonal trendline represents a perfect alignment; data points clustered tightly around it indicate high precision.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 4. FUTURE 30 DAYS FORECAST
    with tab_future:
        st.subheader("Zoomed-In 30-Day Future Projections")
        
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=fore_df["ds"], y=fore_df["Predicted"],
            mode="lines+markers",
            name="30-Day Future Forecast",
            line=dict(color="#F59E0B", width=3)
        ))
        fig_future.add_trace(go.Scatter(
            x=fore_df["ds"], y=fore_df["yhat_upper"],
            mode="lines",
            line=dict(color="rgba(245,158,11,0)"),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig_future.add_trace(go.Scatter(
            x=fore_df["ds"], y=fore_df["yhat_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(245,158,11,0)"),
            name="Confidence Interval Bounds",
            showlegend=True,
            hoverinfo="skip"
        ))
        fig_future.update_layout(
            template="plotly_dark",
            title="30-Day Projections Detail",
            xaxis_title="Forecast Date",
            yaxis_title="Expected Revenue ($)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_future, use_container_width=True)
        
        st.markdown("#### 📋 Detailed Day-by-Day Forecast Output Sheet")
        # Format table
        export_fore = fore_df[["ds", "Predicted", "yhat_lower", "yhat_upper"]].copy()
        export_fore.columns = ["Forecast Date", "Predicted Revenue", "Lower Bound", "Upper Bound"]
        export_fore["Day of Week"] = export_fore["Forecast Date"].dt.day_name()
        export_fore = export_fore[["Forecast Date", "Day of Week", "Predicted Revenue", "Lower Bound", "Upper Bound"]]
        
        st.dataframe(
            export_fore.style.format({
                "Predicted Revenue": "${:,.2f}",
                "Lower Bound": "${:,.2f}",
                "Upper Bound": "${:,.2f}"
            }),
            use_container_width=True
        )
        
        # Dynamic forecast metric
        avg_forecast_val = fore_df["Predicted"].mean()
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💡 Forecast Projection Insight</div>
                <p style="margin:0; color:#E2E8F0;">
                    The mean predicted revenue for the upcoming 30 days is <b>${avg_forecast_val:,.2f}/day</b>, 
                    guiding operational demand planning and inventory management levels.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 5. METRICS & EXPLANATORY SECTION
    with tab_metrics:
        st.subheader("Model Diagnostic Specifications")
        st.markdown(
            f"""
            The forecasting system evaluated models using a **30-day temporal validation split**. 
            Below are the specifications for the winning model:
            - **Model Name**: `{best_model_name}`
            - **RMSE (Root Mean Squared Error)**: `${best_metrics['RMSE']:.2f}` (Penalizes larger forecast anomalies)
            - **MAE (Mean Absolute Error)**: `${best_metrics['MAE']:.2f}` (Average absolute deviation size)
            - **MAPE (Mean Absolute Percentage Error)**: `{best_metrics['MAPE']:.2f}%` (Percentage error relative to daily volumes)
            """
        )
        st.markdown(
            """
            <div class="insight-card" style="background: rgba(99,102,241,0.05); border-color: rgba(99,102,241,0.2);">
                <div class="insight-title">💡 Operational Guidelines</div>
                <p style="margin:0; color:#D1D5DB;">
                    <b>Prophet</b> models fit multi-seasonal patterns (weekly cycles, yearly holiday seasons), while <b>XGBoost</b> excels at non-linear lag-feature mappings. 
                    The system dynamically selects the configuration that delivers the lowest RMSE to safeguard supply chains from stockouts or capital overcommitment.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
