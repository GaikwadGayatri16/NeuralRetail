import streamlit as st
import pandas as pd
from pathlib import Path

# Configure page settings
st.set_page_config(
    page_title="NeuralRetail AI - Intelligent Retail Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (Dark Theme aesthetics, Glassmorphism, Google Fonts)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .main .block-container {
        font-family: 'Outfit', 'Inter', sans-serif;
        padding-top: 1.5rem;
        background-color: #0B0F19;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    /* Premium Title Banner */
    .title-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .title-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 15% 50%, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .title-banner h1 {
        margin: 0;
        font-size: 3rem;
        letter-spacing: -0.025em;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .title-banner p {
        color: #E2E8F0;
        font-size: 1.25rem;
        margin-top: 0.75rem;
        font-weight: 300;
    }
    
    /* Card Styles */
    .custom-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px -5px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
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
    
    /* Team Section Styles */
    .team-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    
    .team-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.25rem;
        color: #F8FAFC;
    }
    
    .team-members {
        color: #94A3B8;
        margin-top: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Icon style */
    .step-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #818CF8;
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

# Title Banner
st.markdown(
    """
    <div class="title-banner">
        <h1>🛍️ NeuralRetail AI</h1>
        <p>Enterprise Retail Intelligence & Customer Analytics Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Data availability check
BASE_DIR = Path(__file__).resolve().parent
cleaned_csv_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"

# KPI metrics header if data is available
if cleaned_csv_path.exists():
    try:
        # Load simple stats from metadata or cached info
        @st.cache_data
        def load_summary_stats(path):
            df = pd.read_csv(path, usecols=["Customer ID", "StockCode", "TotalPrice", "InvoiceDate"])
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            total_rev = df["TotalPrice"].sum()
            num_cust = df["Customer ID"].nunique()
            num_skus = df["StockCode"].nunique()
            min_date = df["InvoiceDate"].min().strftime('%b %Y')
            max_date = df["InvoiceDate"].max().strftime('%b %Y')
            return len(df), total_rev, num_cust, num_skus, f"{min_date} - {max_date}"
            
        row_count, revenue, customers, skus, date_range = load_summary_stats(cleaned_csv_path)
        
        # Draw metric cards on top
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        draw_kpi_card(col_m1, "Total Transactions", f"{row_count:,}", "📊")
        draw_kpi_card(col_m2, "Total Revenue", f"${revenue:,.2f}", "💰")
        draw_kpi_card(col_m3, "Unique Customers", f"{customers:,}", "👥")
        draw_kpi_card(col_m4, "Active SKUs", f"{skus:,}", "📦")
        draw_kpi_card(col_m5, "Timeline Scope", date_range, "📅")
        st.divider()
    except Exception:
        pass

# Project Overview & Quick Stats
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(
        """
        <div class="custom-card">
            <h3>Enterprise Project Summary</h3>
            <p>
                <b>NeuralRetail AI</b> is a state-of-the-art academic retail intelligence platform developed by a 
                collaborative team of 6 students. Using machine learning and statistical modeling on the historical 
                <b>Online Retail II dataset</b>, this platform provides actionable insights for transaction trends, 
                customer clusters, churn prediction, demand forecasting, and inventory optimizations.
            </p>
            <p>
                Use the sidebar to navigate through the executive analytics dashboards, where insights are categorized 
                by focus group and pipeline objectives.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 📊 Project Focus Areas")
    
    obj_col1, obj_col2 = st.columns(2)
    with obj_col1:
        st.markdown(
            """
            <div class="custom-card">
                <div class="step-icon">👥</div>
                <h4>Customer Intelligence & Segmentation</h4>
                <p>Segment shoppers via RFM modeling to identify VIPs and target at-risk groups using KMeans, DBSCAN, and GMM algorithms.</p>
            </div>
            <div class="custom-card">
                <div class="step-icon">⚠️</div>
                <h4>Predictive Churn Analytics</h4>
                <p>Flag customers likely to churn using classification models (Logistic Regression, Random Forest, XGBoost) trained on temporal cuts.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with obj_col2:
        st.markdown(
            """
            <div class="custom-card">
                <div class="step-icon">📈</div>
                <h4>Demand Forecasting</h4>
                <p>Estimate future daily transaction values utilizing Prophet and XGBoost time-series models for sales planning.</p>
            </div>
            <div class="custom-card">
                <div class="step-icon">📦</div>
                <h4>Inventory Control & Optimization</h4>
                <p>Run ABC Analysis and automate reorder planning (ROP, Safety Stock, EOQ) to keep supply chain levels optimized.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_right:
    st.markdown("### 👥 Project Team Structure")
    
    st.markdown(
        """
        <div class="custom-card">
            <span class="team-badge">Group 1</span>
            <div class="team-title">Customer Intelligence Team</div>
            <div class="team-members">
                • RFM Analysis & Profiling<br>
                • Clustering Models Comparison<br>
                • Customer Insights & Personas
            </div>
        </div>
        
        <div class="custom-card">
            <span class="team-badge" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border-color: rgba(16, 185, 129, 0.25);">Group 2</span>
            <div class="team-title">Churn Analytics Team</div>
            <div class="team-members">
                • Churn Classification Modeling<br>
                • Evaluation & Feature Importance<br>
                • High Churn Risk Identification
            </div>
        </div>
        
        <div class="custom-card">
            <span class="team-badge" style="background: rgba(245, 158, 11, 0.15); color: #FBBF24; border-color: rgba(245, 158, 11, 0.25);">Group 3</span>
            <div class="team-title">Forecast & Inventory Team</div>
            <div class="team-members">
                • Time-Series Demand Forecasting<br>
                • ABC Inventory Categorization<br>
                • Automated Reorder Calculations (ROP/EOQ)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# File status and data overview section
st.markdown("### 🗄️ Dataset Preview")

if cleaned_csv_path.exists():
    try:
        # Load a small sample to avoid memory pressure
        df_preview = pd.read_csv(cleaned_csv_path, nrows=5)
        st.success("✅ Cleaned retail dataset successfully loaded.")
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Cleaned dataset found but failed to load preview: {e}")
else:
    st.info("ℹ️ Dataset files have not been generated yet. Please run the data pipelines to generate files.")
