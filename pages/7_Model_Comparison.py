import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Model Comparison - NeuralRetail AI", page_icon="⚖️", layout="wide")

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
    
    .winning-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 1.75rem;
        margin-top: 1.25rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .winning-card:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
    }
    
    .winning-title {
        color: #34D399;
        font-weight: 700;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
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
        <h1 class="hero-title">🏆 Model Comparison Leaderboard</h1>
        <p class="hero-desc">Consolidated model evaluation dashboard. Compare algorithm metrics, error distributions, and identify winner models.</p>
        <span class="hero-purpose">🎯 Business Purpose: Benchmark machine learning performance across segmentation, classification, and forecasting pipelines.</span>
    </div>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parents[1]
metrics_path = BASE_DIR / "outputs" / "model_comparison.json"

if not metrics_path.exists():
    st.warning("⚠️ Consolidated metrics report not found. Please run the full evaluation pipeline (`src/evaluation.py`) to generate comparison metrics.")
else:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    # Pre-read winners
    best_seg = metrics["segmentation"]["best_model"]
    best_churn = metrics["churn"]["best_model"]
    best_fore = metrics["forecasting"]["best_model"]
    
    # KPI metrics cards on top
    col1, col2, col3, col4 = st.columns(4)
    draw_kpi_card(col1, "Segmentation Winner", best_seg, "👥")
    draw_kpi_card(col2, "Churn Model Winner", best_churn, "⚠️")
    draw_kpi_card(col3, "Forecasting Winner", best_fore, "📈")
    draw_kpi_card(col4, "Consolidation Status", "✅ Completed", "📊")
    
    st.divider()
    
    # Create Tabs
    tab_seg, tab_churn, tab_fore, tab_best = st.tabs([
        "👥 Segmentation Models",
        "⚠️ Churn Models",
        "📈 Forecasting Models",
        "🏆 Best Overall Models"
    ])
    
    # 1. SEGMENTATION MODELS TAB
    with tab_seg:
        st.subheader("Customer Segmentation Algorithm Benchmarking")
        st.write("Performance compared using **Silhouette Score** (higher is better, indicating cluster separation) and **Davies-Bouldin Index** (lower is better, indicating cluster density).")
        
        seg_data = metrics["segmentation"]["metrics"]
        
        seg_list = []
        for model_name, m_dict in seg_data.items():
            row = {
                "Algorithm": model_name,
                "Silhouette Score": m_dict["Silhouette"],
                "Davies-Bouldin Index": m_dict["Davies-Bouldin"],
                "Selected Best": "⭐ Winner" if model_name == best_seg else ""
            }
            seg_list.append(row)
            
        df_seg_metrics = pd.DataFrame(seg_list)
        st.dataframe(
            df_seg_metrics.style.format({
                "Silhouette Score": "{:.4f}",
                "Davies-Bouldin Index": "{:.4f}"
            }),
            use_container_width=True
        )
        
        # Highlight winner using st.success
        st.success(f"🏆 **Winning Segmentation Model:** `{best_seg}` (highest Silhouette Score of {seg_data[best_seg]['Silhouette']:.4f}).")
        
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Segmentation Accuracy</div>
                <p style="margin:0; color:#E2E8F0;">
                    `{best_seg}` achieves the optimal balance between high intra-cluster similarity and distinct inter-cluster separation, 
                    ensuring that marketing personas correspond to distinct customer behaviors.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 2. CHURN MODELS TAB
    with tab_churn:
        st.subheader("Churn Classifier Benchmarking")
        st.write("Performance compared using **Accuracy**, **Precision**, **Recall**, **F1-Score** (harmonic mean of Precision/Recall), and **ROC AUC** (Receiver Operating Characteristic Area Under Curve).")
        
        churn_data = metrics["churn"]["metrics"]
        
        churn_list = []
        for model_name, m_dict in churn_data.items():
            row = {
                "Classifier": model_name,
                "Accuracy": m_dict["Accuracy"],
                "Precision": m_dict["Precision"],
                "Recall": m_dict["Recall"],
                "F1-Score": m_dict["F1 Score"],
                "ROC AUC": m_dict["ROC AUC"],
                "Selected Best": "⭐ Winner" if model_name == best_churn else ""
            }
            churn_list.append(row)
            
        df_churn_metrics = pd.DataFrame(churn_list)
        st.dataframe(
            df_churn_metrics.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "ROC AUC": "{:.4f}"
            }),
            use_container_width=True
        )
        
        # Highlight winner using st.success
        best_f1 = churn_data[best_churn]['F1 Score']
        st.success(f"🏆 **Winning Churn Model:** `{best_churn}` (highest F1-Score of {best_f1:.2%} on validation temporal split).")
        
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Classification Metrics</div>
                <p style="margin:0; color:#E2E8F0;">
                    F1-Score was chosen as the primary selection metric to strike a business balance between false positives (marketing cost leakages) 
                    and false negatives (unflagged churned customers). `{best_churn}` provides the most robust predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 3. FORECASTING MODELS TAB
    with tab_fore:
        st.subheader("Demand Forecasting Model Benchmarking")
        st.write("Performance compared using **Root Mean Squared Error (RMSE)** (penalizes large errors, lower is better), **Mean Absolute Error (MAE)** (average error size, lower is better), and **Mean Absolute Percentage Error (MAPE)** (relative error scale, lower is better).")
        
        fore_data = metrics["forecasting"]["metrics"]
        
        fore_list = []
        for model_name, m_dict in fore_data.items():
            row = {
                "Forecaster": model_name,
                "RMSE": m_dict["RMSE"],
                "MAE": m_dict["MAE"],
                "MAPE": m_dict["MAPE"],
                "Selected Best": "⭐ Winner" if model_name == best_fore else ""
            }
            fore_list.append(row)
            
        df_fore_metrics = pd.DataFrame(fore_list)
        st.dataframe(
            df_fore_metrics.style.format({
                "RMSE": "${:,.2f}",
                "MAE": "${:,.2f}",
                "MAPE": "{:.2f}%"
            }),
            use_container_width=True
        )
        
        # Highlight winner using st.success
        st.success(f"🏆 **Winning Forecasting Model:** `{best_fore}` (lowest RMSE of ${fore_data[best_fore]['RMSE']:.2f} on 30-day temporal test split).")
        
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Forecasting Metrics</div>
                <p style="margin:0; color:#E2E8F0;">
                    The winning model `{best_fore}` exhibits the lowest root mean squared error, meaning it is least prone to massive seasonal 
                    prediction errors, which is critical for supply chain stability.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 4. BEST OVERALL MODELS TAB
    with tab_best:
        st.subheader("System Architecture & Winning Algorithms")
        st.write("These winning configurations are serialized and dynamically integrated into the dashboards to power downstream analytics.")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.markdown(
                f"""
                <div class="winning-card">
                    <div class="winning-title">👥 Customer Segmentation</div>
                    <h3 style="margin-top:0.5rem; margin-bottom:0.25rem;">{best_seg}</h3>
                    <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:0.75rem;">Group 1 Primary Model</p>
                    <div style="font-size:0.95rem;">
                        • Silhouette: <b>{seg_data[best_seg]['Silhouette']:.4f}</b><br>
                        • Davies-Bouldin: <b>{seg_data[best_seg]['Davies-Bouldin']:.4f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_c2:
            st.markdown(
                f"""
                <div class="winning-card" style="border-color: rgba(245,158,11,0.3); background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(245,158,11,0.05) 100%);">
                    <div class="winning-title" style="color:#FBBF24;">⚠️ Churn Prediction</div>
                    <h3 style="margin-top:0.5rem; margin-bottom:0.25rem;">{best_churn}</h3>
                    <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:0.75rem;">Group 2 Primary Model</p>
                    <div style="font-size:0.95rem;">
                        • F1-Score: <b>{churn_data[best_churn]['F1 Score']:.2%}</b><br>
                        • ROC AUC: <b>{churn_data[best_churn]['ROC AUC']:.4f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_c3:
            st.markdown(
                f"""
                <div class="winning-card" style="border-color: rgba(99,102,241,0.3); background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(99,102,241,0.05) 100%);">
                    <div class="winning-title" style="color:#818CF8;">📈 Demand Forecasting</div>
                    <h3 style="margin-top:0.5rem; margin-bottom:0.25rem;">{best_fore}</h3>
                    <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:0.75rem;">Group 3 Primary Model</p>
                    <div style="font-size:0.95rem;">
                        • RMSE: <b>${fore_data[best_fore]['RMSE']:.2f}</b><br>
                        • MAPE: <b>{fore_data[best_fore]['MAPE']:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

