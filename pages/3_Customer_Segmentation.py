import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Customer Segmentation - NeuralRetail AI", page_icon="👥", layout="wide")

# Custom premium CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
        background-color: #0B0F19;
    }
    
    /* Hero Section Styles */
    .hero-section {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
    }
    .hero-desc {
        color: #E2E8F0;
        font-size: 1.1rem;
        margin: 0 0 0.75rem 0;
        font-weight: 300;
    }
    .hero-purpose {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* KPI Card Styles */
    .kpi-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(30, 41, 59, 0.55) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.03), transparent);
        transform: translateX(-100%);
        transition: transform 0.5s ease;
    }
    .kpi-card:hover::before {
        transform: translateX(100%);
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 25px -5px rgba(99, 102, 241, 0.2);
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-icon {
        font-size: 1.4rem;
        color: #818CF8;
        background: rgba(99, 102, 241, 0.12);
        padding: 0.3rem 0.5rem;
        border-radius: 8px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
        margin-top: 0.25rem;
    }
    
    .segment-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .vip-tag { color: #F59E0B; font-weight: bold; }
    .loyal-tag { color: #10B981; font-weight: bold; }
    .regular-tag { color: #3B82F6; font-weight: bold; }
    .risk-tag { color: #EF4444; font-weight: bold; }
    
    /* Insight Card Styles */
    .insight-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
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
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def draw_kpi_card(col, title, value, icon):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">{title}</span>
            <span class="kpi-icon">{icon}</span>
        </div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# Hero Section
st.markdown(
    """
    <div class="hero-section">
        <h1 class="hero-title">👥 Customer Segmentation (RFM)</h1>
        <p class="hero-desc">Profile customer behavior segments based on Recency, Frequency, and Monetary metrics.</p>
        <span class="hero-purpose">🎯 Business Purpose: Classify customers into target segments (VIP, Loyal, Regular, At Risk) to tailor custom marketing strategies.</span>
    </div>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parents[1]
segments_path = BASE_DIR / "outputs" / "customer_segments.csv"

if not segments_path.exists():
    st.warning("⚠️ Segmented customer profile outputs not found. Please run the Customer Segmentation pipeline (`src/segmentation.py`) to generate this data.")
else:
    @st.cache_data
    def load_segments(path):
        return pd.read_csv(path)
        
    df_seg = load_segments(segments_path)
    
    # Calculate counts and percentages dynamically
    total_cust = len(df_seg)
    segment_counts = df_seg["Segment"].value_counts()
    
    vip_count = segment_counts.get("VIP Customers", 0)
    loyal_count = segment_counts.get("Loyal Customers", 0)
    regular_count = segment_counts.get("Regular Customers", 0)
    risk_count = segment_counts.get("At Risk Customers", 0)
    
    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    draw_kpi_card(col1, f"VIP Shoppers ({(vip_count/total_cust*100):.1f}%)", f"{vip_count:,}", "👑")
    draw_kpi_card(col2, f"Loyal Customers ({(loyal_count/total_cust*100):.1f}%)", f"{loyal_count:,}", "🤝")
    draw_kpi_card(col3, f"Regular Shoppers ({(regular_count/total_cust*100):.1f}%)", f"{regular_count:,}", "🛒")
    draw_kpi_card(col4, f"At Risk Customers ({(risk_count/total_cust*100):.1f}%)", f"{risk_count:,}", "⚠️")
    
    st.divider()
    
    # Create Tabs
    tab_dist, tab_rfm, tab_vip, tab_cluster, tab_stats = st.tabs([
        "📊 Segment Distribution",
        "⚖️ RFM Analysis",
        "👑 VIP Customers List",
        "🌐 Cluster Visualization",
        "📋 Segment Statistics & Strategy"
    ])
    
    # 1. SEGMENT DISTRIBUTION
    with tab_dist:
        st.subheader("Customer Segment Shares")
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
            },
            hole=0.4
        )
        fig_pie.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Dynamic Insight
        top_segment_name = counts.iloc[0]["Segment"]
        top_segment_pct = (counts.iloc[0]["Count"] / total_cust) * 100
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Segment Distribution</div>
                <p style="margin:0; color:#E2E8F0;">
                    The largest segment group is <b>{top_segment_name}</b>, representing <b>{top_segment_pct:.2f}%</b> 
                    of the total customer base ({counts.iloc[0]['Count']:,} shoppers).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 2. RFM ANALYSIS
    with tab_rfm:
        st.subheader("RFM Component Analysis Across Segments")
        
        rfm_metric = st.selectbox(
            "Select RFM Metric to Plot", 
            ["Monetary ($ Value)", "Frequency (Order Count)", "Recency (Days Since Last Order)"]
        )
        
        if rfm_metric == "Monetary ($ Value)":
            fig_box = px.box(
                df_seg, x="Segment", y="Monetary", color="Segment",
                title="Customer Monetary Value Comparison ($ Log Scale)",
                log_y=True,
                color_discrete_map={
                    "VIP Customers": "#F59E0B",
                    "Loyal Customers": "#10B981",
                    "Regular Customers": "#3B82F6",
                    "At Risk Customers": "#EF4444"
                }
            )
            val_description = "monetary value (spending capacity)"
        elif rfm_metric == "Frequency (Order Count)":
            fig_box = px.box(
                df_seg, x="Segment", y="Frequency", color="Segment",
                title="Customer Order Frequency Comparison (Log Scale)",
                log_y=True,
                color_discrete_map={
                    "VIP Customers": "#F59E0B",
                    "Loyal Customers": "#10B981",
                    "Regular Customers": "#3B82F6",
                    "At Risk Customers": "#EF4444"
                }
            )
            val_description = "purchase frequency (total orders)"
        else:
            fig_box = px.box(
                df_seg, x="Segment", y="Recency", color="Segment",
                title="Customer Recency Comparison (Days Since Last Purchase)",
                color_discrete_map={
                    "VIP Customers": "#F59E0B",
                    "Loyal Customers": "#10B981",
                    "Regular Customers": "#3B82F6",
                    "At Risk Customers": "#EF4444"
                }
            )
            val_description = "days since last purchase (retention activity)"
            
        fig_box.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Dynamic Insight
        vip_avg_val = df_seg[df_seg["Segment"] == "VIP Customers"]["Monetary"].mean()
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: RFM Metric Analysis</div>
                <p style="margin:0; color:#E2E8F0;">
                    VIP Customers exhibit an average spending power of <b>${vip_avg_val:,.2f}</b>, 
                    validating their status as high-value assets. Boxplots utilize logarithmic scaling to account for extreme customer outliers.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 3. VIP CUSTOMERS LIST
    with tab_vip:
        st.subheader("VIP Customers Register (High-Priority Target List)")
        vip_df = df_seg[df_seg["Segment"] == "VIP Customers"].sort_values(by="Monetary", ascending=False)
        
        st.markdown(
            f"""
            Below is the complete register of <b>{len(vip_df):,} VIP Customers</b>. These customers shop regularly, spend heavily, and have high recent engagements.
            """
        )
        
        st.dataframe(
            vip_df[["Customer ID", "Recency", "Frequency", "Monetary"]].style.format({
                "Monetary": "${:,.2f}",
                "Recency": "{:.0f} days",
                "Frequency": "{:.0f} orders"
            }),
            use_container_width=True
        )
        
        # Dynamic Insight
        vip_total_spend = vip_df["Monetary"].sum()
        grand_total_spend = df_seg["Monetary"].sum()
        vip_spend_pct = (vip_total_spend / grand_total_spend * 100) if grand_total_spend > 0 else 0
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: VIP Contribution</div>
                <p style="margin:0; color:#E2E8F0;">
                    Although VIP customers represent only <b>{(vip_count/total_cust*100):.2f}%</b> of the shopper count, they contribute 
                    <b>{vip_spend_pct:.2f}%</b> of total overall revenue (amounting to <b>${vip_total_spend:,.2f}</b>).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 4. CLUSTER VISUALIZATION
    with tab_cluster:
        st.subheader("3D Interactive Cluster Mapping")
        
        use_log = st.checkbox("Apply Log Scale (Highly recommended for visual clustering clarity)", value=True)
        
        viz_df = df_seg.copy()
        if use_log:
            viz_df["Recency"] = np.log1p(viz_df["Recency"])
            viz_df["Frequency"] = np.log1p(viz_df["Frequency"])
            viz_df["Monetary"] = np.log1p(viz_df["Monetary"])
            
        fig_3d = px.scatter_3d(
            viz_df, x="Recency", y="Frequency", z="Monetary",
            color="Segment",
            opacity=0.7,
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
        
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Spatial Separation</div>
                <p style="margin:0; color:#E2E8F0;">
                    The 3D space maps clusters along the Recency (X), Frequency (Y), and Monetary (Z) axes. Clear separation demonstrates 
                    the model's capability to categorize customer profiles logically (e.g. VIPs are clustered high on Frequency & Monetary, and low on Recency).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 5. SEGMENT STATISTICS
    with tab_stats:
        st.subheader("Segment Profiles & Marketing Playbooks")
        
        # Segment profile matrix table
        summary = df_seg.groupby("Segment").agg(
            CustomerCount=("Customer ID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean")
        ).round(1).reset_index()
        
        st.dataframe(
            summary.style.format({
                "AvgMonetary": "${:,.2f}",
                "CustomerCount": "{:,}",
                "AvgRecency": "{:.1f} days",
                "AvgFrequency": "{:.1f} orders"
            }),
            use_container_width=True
        )
        
        st.divider()
        
        st.subheader("💡 Strategic Marketing Playbooks")
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

