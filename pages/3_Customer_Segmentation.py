import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Customer Segmentation - NeuralRetail AI", page_icon="👥", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .segment-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .vip-tag {
        color: #FBBF24;
        font-weight: bold;
    }
    .loyal-tag {
        color: #34D399;
        font-weight: bold;
    }
    .regular-tag {
        color: #60A5FA;
        font-weight: bold;
    }
    .risk-tag {
        color: #F87171;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👥 Customer Segmentation (RFM)")
st.markdown("Group 1 Customer Intelligence Dashboard. Understand customer behaviors, lifetime values, and segments.")

BASE_DIR = Path(__file__).resolve().parents[1]
segments_path = BASE_DIR / "outputs" / "customer_segments.csv"

if not segments_path.exists():
    st.warning("⚠️ Segmented customer profile outputs not found. Please run the Customer Segmentation pipeline (`src/segmentation.py`) to generate this data.")
else:
    @st.cache_data
    def load_segments(path):
        return pd.read_csv(path)
        
    df_seg = load_segments(segments_path)
    
    # 1. High-level counts
    st.markdown("### 📊 Segment Statistics")
    
    col_stat1, col_stat2 = st.columns([1, 2])
    
    with col_stat1:
        counts = df_seg["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Count"]
        
        fig_pie = px.pie(
            counts, names="Segment", values="Count",
            title="Customer Distribution by Segment",
            color="Segment",
            color_discrete_map={
                "VIP Customers": "#F59E0B",
                "Loyal Customers": "#10B981",
                "Regular Customers": "#3B82F6",
                "At Risk Customers": "#EF4444"
            }
        )
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_stat2:
        # Show segment metrics summary
        summary = df_seg.groupby("Segment").agg(
            CustomerCount=("Customer ID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean")
        ).round(1).reset_index()
        
        st.write("#### 📋 Segment Profile Matrix")
        st.dataframe(
            summary.style.format({
                "AvgMonetary": "${:,.2f}",
                "CustomerCount": "{:,}",
                "AvgRecency": "{:.1f} days",
                "AvgFrequency": "{:.1f} orders"
            }),
            use_container_width=True
        )
        
    # 2. 3D Cluster Visualizations
    st.markdown("### 🌐 3D Customer Clusters Scatter Plot")
    
    # Let user toggle log scale for better visualization spacing
    use_log = st.checkbox("Apply Log Scale to Axes (Recommended for viewing skewed distributions)", value=True)
    
    viz_df = df_seg.copy()
    if use_log:
        viz_df["Recency"] = np.log1p(viz_df["Recency"])
        viz_df["Frequency"] = np.log1p(viz_df["Frequency"])
        viz_df["Monetary"] = np.log1p(viz_df["Monetary"])
        
    fig_3d = px.scatter_3d(
        viz_df, x="Recency", y="Frequency", z="Monetary",
        color="Segment",
        opacity=0.6,
        title="3D RFM Cluster Distribution",
        labels={
            "Recency": "Recency (Log)" if use_log else "Recency (Days)",
            "Frequency": "Frequency (Log)" if use_log else "Frequency (Orders)",
            "Monetary": "Monetary (Log)" if use_log else "Monetary ($)"
        },
        color_discrete_map={
            "VIP Customers": "#F59E0B",
            "Loyal Customers": "#10B981",
            "Regular Customers": "#3B82F6",
            "At Risk Customers": "#EF4444"
        }
    )
    fig_3d.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # 3. Business Personas
    st.markdown("### 💡 Marketing Personas & Strategy")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown(
            """
            <div class="segment-card">
                <h4>👑 <span class="vip-tag">VIP Customers</span></h4>
                <p><b>Characteristics:</b> Shop frequently, buy high-value items, and have purchased very recently.</p>
                <p><b>Strategy:</b> Enroll in premium loyalty programs, offer exclusive early access to new product releases, and provide dedicated customer support.</p>
            </div>
            
            <div class="segment-card">
                <h4>🤝 <span class="loyal-tag">Loyal Customers</span></h4>
                <p><b>Characteristics:</b> High frequency and moderate-to-high spending. Recency is low.</p>
                <p><b>Strategy:</b> Target with personalized upselling/cross-selling, offer referral bonuses, and send thank-you gifts.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_p2:
        st.markdown(
            """
            <div class="segment-card">
                <h4>🛒 <span class="regular-tag">Regular Customers</span></h4>
                <p><b>Characteristics:</b> Average shopping behavior. Recent purchase, but lower transaction counts.</p>
                <p><b>Strategy:</b> Offer bundle deals, run standard seasonal promotions, and encourage higher order quantities.</p>
            </div>
            
            <div class="segment-card">
                <h4>⚠️ <span class="risk-tag">At Risk Customers</span></h4>
                <p><b>Characteristics:</b> Haven't purchased in a long time. Low lifetime orders and spend.</p>
                <p><b>Strategy:</b> Launch automated Win-Back email campaigns, provide steep discount coupons, and solicit feedback on past orders.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
