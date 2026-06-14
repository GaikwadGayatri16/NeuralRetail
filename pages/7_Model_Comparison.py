import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Model Comparison - NeuralRetail AI", page_icon="⚖️", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .best-badge {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .model-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-table {
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚖️ Model Comparison")
st.markdown("Consolidated model evaluation dashboard. Compare algorithm metrics and check winning models.")

BASE_DIR = Path(__file__).resolve().parents[1]
metrics_path = BASE_DIR / "outputs" / "model_comparison.json"

if not metrics_path.exists():
    st.warning("⚠️ Consolidated metrics report not found. Please run the full evaluation pipeline (`src/evaluation.py`) to generate comparison metrics.")
else:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    # --------------------------
    # 1. CUSTOMER SEGMENTATION
    # --------------------------
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("### 👥 Customer Segmentation Algorithms")
    st.markdown("Performance compared using **Silhouette Score** (higher is better, indicating cluster separation) and **Davies-Bouldin Index** (lower is better, indicating cluster density).")
    
    seg_data = metrics["segmentation"]["metrics"]
    best_seg = metrics["segmentation"]["best_model"]
    
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
    st.markdown(f"**Winning Segmentation Model:** `{best_seg}` (highest Silhouette Score).", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --------------------------
    # 2. CHURN PREDICTION
    # --------------------------
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Churn Prediction Classifiers")
    st.markdown("Performance compared using **Accuracy**, **Precision**, **Recall**, **F1-Score** (harmonic mean of Precision/Recall), and **ROC AUC** (Receiver Operating Characteristic Area Under Curve).")
    
    churn_data = metrics["churn"]["metrics"]
    best_churn = metrics["churn"]["best_model"]
    
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
    st.markdown(f"**Winning Churn Model:** `{best_churn}` (highest F1-Score on test split).", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --------------------------
    # 3. DEMAND FORECASTING
    # --------------------------
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Demand Forecasting Models")
    st.markdown("Performance compared using **Root Mean Squared Error (RMSE)** (penalizes large errors, lower is better), **Mean Absolute Error (MAE)** (average error size, lower is better), and **Mean Absolute Percentage Error (MAPE)** (relative error scale, lower is better).")
    
    fore_data = metrics["forecasting"]["metrics"]
    best_fore = metrics["forecasting"]["best_model"]
    
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
    st.markdown(f"**Winning Forecasting Model:** `{best_fore}` (lowest RMSE on test split).", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
