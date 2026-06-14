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
        padding-top: 2rem;
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
        margin-bottom: 2.5rem;
        text-align: center;
    }
    
    .title-banner h1 {
        margin: 0;
        font-size: 3rem;
        letter-spacing: -0.025em;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .title-banner p {
        color: #E2E8F0;
        font-size: 1.25rem;
        margin-top: 0.75rem;
        font-weight: 300;
    }
    
    /* Card Styles */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .custom-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -8px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Team Section Styles */
    .team-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .team-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 0.5rem;
        color: #F8FAFC;
    }
    
    .team-members {
        color: #94A3B8;
        margin-top: 0.5rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Icon style */
    .step-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Banner
st.markdown(
    """
    <div class="title-banner">
        <h1>🛍️ NeuralRetail AI</h1>
        <p>Intelligent Retail Analytics & Customer Intelligence Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Project Overview & Quick Stats
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(
        """
        <div class="custom-card">
            <h3>Project Description</h3>
            <p>
                <b>NeuralRetail AI</b> is a state-of-the-art academic retail intelligence platform developed by a 
                collaborative team of 6 students. Using machine learning and statistical modeling on the historical 
                <b>Online Retail II dataset</b>, this platform provides actionable insights for transaction trends, 
                customer clusters, churn prediction, demand forecasting, and inventory optimizations.
            </p>
            <p>
                Navigate through the sidebar to explore in-depth dashboards corresponding to the responsibilities 
                of our three specialized sub-teams.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 📊 Project Objectives")
    
    obj_col1, obj_col2 = st.columns(2)
    with obj_col1:
        st.markdown(
            """
            <div class="custom-card">
                <div class="step-icon">🎯</div>
                <h4>Customer Intelligence</h4>
                <p>Segment shoppers via RFM modeling to identify VIPs and target at-risk groups using KMeans, DBSCAN, and GMM.</p>
            </div>
            <div class="custom-card">
                <div class="step-icon">⏱️</div>
                <h4>Predictive Churn</h4>
                <p>Flag customers likely to drift away using advanced classification models trained on historical intervals.</p>
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
                <p>Estimate future daily transaction values utilizing Prophet and XGBoost time-series models.</p>
            </div>
            <div class="custom-card">
                <div class="step-icon">📦</div>
                <h4>Inventory Control</h4>
                <p>Run ABC Analysis and automate reorder planning (ROP, Safety Stock, EOQ) to keep shelves optimally stocked.</p>
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
            <span class="team-badge" style="background: rgba(16, 185, 129, 0.2); color: #34D399; border-color: rgba(16, 185, 129, 0.3);">Group 2</span>
            <div class="team-title">Churn Analytics Team</div>
            <div class="team-members">
                • Churn Classification Modeling<br>
                • Evaluation & Feature Importance<br>
                • High Churn Risk Identification
            </div>
        </div>
        
        <div class="custom-card">
            <span class="team-badge" style="background: rgba(245, 158, 11, 0.2); color: #FBBF24; border-color: rgba(245, 158, 11, 0.3);">Group 3</span>
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
st.markdown("### 🗄️ Dataset Quick Overview")

# Try to load cleaned dataset for preview
BASE_DIR = Path(__file__).resolve().parent
cleaned_csv_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"

if cleaned_csv_path.exists():
    try:
        # Load a small sample to avoid memory pressure
        df_preview = pd.read_csv(cleaned_csv_path, nrows=5)
        st.success("✅ Cleaned retail dataset successfully loaded and cached.")
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Cleaned dataset found but failed to load preview: {e}")
else:
    st.info("ℹ️ Dataset files have not been generated yet. Please run the data pipelines to generate files.")
