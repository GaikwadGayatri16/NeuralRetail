import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set page settings
st.set_page_config(page_title="Churn Prediction - NeuralRetail AI", page_icon="⚠️", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .search-panel {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-panel {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .high-risk-badge {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .medium-risk-badge {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .low-risk-badge {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚠️ Churn Prediction")
st.markdown("Group 2 Churn Analytics Dashboard. Predict customer churn probability and analyze risk factors.")

BASE_DIR = Path(__file__).resolve().parents[1]
predictions_path = BASE_DIR / "outputs" / "churn_predictions.csv"
model_path = BASE_DIR / "models" / "best_churn_model.pkl"

if not predictions_path.exists() or not model_path.exists():
    st.warning("⚠️ Churn prediction outputs or models not found. Please run the Churn Prediction pipeline (`src/churn.py`) to generate this data.")
else:
    @st.cache_data
    def load_churn_data(pred_path):
        return pd.read_csv(pred_path)
        
    @st.cache_resource
    def load_churn_model(mod_path):
        return joblib.load(mod_path)
        
    df_churn = load_churn_data(predictions_path)
    model_dict = load_churn_model(model_path)
    
    # Clean customer IDs to strings for easy search
    df_churn["Customer ID"] = df_churn["Customer ID"].astype(str)
    
    # 1. Interactive Customer Search
    st.markdown("### 🔍 Customer Risk Lookup")
    
    with st.container():
        st.markdown('<div class="search-panel">', unsafe_allow_html=True)
        search_id = st.text_input("Enter Customer ID to assess churn risk:", placeholder="e.g. 17850")
        
        if search_id:
            # Match customer
            cust_row = df_churn[df_churn["Customer ID"] == search_id.strip()]
            
            if cust_row.empty:
                st.error(f"Customer ID `{search_id}` not found in the database.")
            else:
                row = cust_row.iloc[0]
                prob = row["ChurnProbability"]
                
                # Assign risk category
                if prob >= 0.70:
                    risk_badge = '<span class="high-risk-badge">High Risk (🚨)</span>'
                elif prob >= 0.30:
                    risk_badge = '<span class="medium-risk-badge">Medium Risk (⚠️)</span>'
                else:
                    risk_badge = '<span class="low-risk-badge">Low Risk (✅)</span>'
                    
                st.markdown(f"#### Customer Churn Risk Level: {risk_badge}", unsafe_allow_html=True)
                
                # Show key stats
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    st.markdown(f'<div class="metric-panel"><b>Recency</b><br>{int(row["Recency"])} days</div>', unsafe_allow_html=True)
                with col_c2:
                    st.markdown(f'<div class="metric-panel"><b>Frequency</b><br>{int(row["Frequency"])} orders</div>', unsafe_allow_html=True)
                with col_c3:
                    st.markdown(f'<div class="metric-panel"><b>Monetary</b><br>${row["Monetary"]:,.2f}</div>', unsafe_allow_html=True)
                with col_c4:
                    st.markdown(f'<div class="metric-panel"><b>Lifetime Value (LTV)</b><br>${row["LifetimeRevenue"]:,.2f}</div>', unsafe_allow_html=True)
                    
                st.markdown(f"**Customer Age (Days since first purchase):** {int(row['CustomerAgeDays'])} days")
                st.markdown(f"**Days Since Last Purchase:** {int(row['DaysSinceLastPurchase'])} days")
                st.markdown(f"**Average Order Value:** ${row['AverageOrderValue']:.2f}")
                
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 2. General Churn Analytics
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("### 📊 Churn Probability Distribution")
        fig_dist = px.histogram(
            df_churn, x="ChurnProbability", nbins=50,
            title="Distribution of Churn Probabilities",
            labels={"ChurnProbability": "Churn Probability"},
            color_discrete_sequence=["#EF4444"]
        )
        fig_dist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_g2:
        # Display Feature Importance
        st.markdown("### 🔑 Model Feature Importance")
        
        # Get importances from model object
        model_obj = model_dict["model"]
        features = model_dict["features"]
        
        importances = None
        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
        elif hasattr(model_obj, "coef_"):
            importances = np.abs(model_obj.coef_[0])
            
        if importances is not None:
            feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
            feat_df = feat_df.sort_values(by="Importance", ascending=True)
            
            fig_feat = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                title=f"Feature Importance ({model_dict['model_name']})",
                color="Importance", color_continuous_scale="Reds"
            )
            fig_feat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
            
    # 3. Target Marketing: List of high risk customers
    st.markdown("### 🎯 High Churn Risk Target Group (Top 100)")
    # Filter customers who are active but have high churn probability (exclude those who already have ground truth churn, i.e. Recency > 90)
    high_risk_list = df_churn[
        (df_churn["ChurnProbability"] >= 0.70) & 
        (df_churn["DaysSinceLastPurchase"] <= 90)
    ].sort_values(by="ChurnProbability", ascending=False).head(100)
    
    if high_risk_list.empty:
        # If none fit this strict criteria, just show the highest probability active customers
        high_risk_list = df_churn[df_churn["DaysSinceLastPurchase"] <= 90].sort_values(by="ChurnProbability", ascending=False).head(100)
        
    st.markdown("These customers have purchased within the last 90 days, but show high probabilities of churning based on their transaction patterns (low frequency, dropping spend, etc.). **Target them with campaigns immediately!**")
    st.dataframe(
        high_risk_list[[
            "Customer ID", "Recency", "Frequency", "Monetary", "AverageOrderValue", "ChurnProbability"
        ]].style.format({
            "ChurnProbability": "{:.2%}",
            "Monetary": "${:,.2f}",
            "AverageOrderValue": "${:,.2f}",
            "Recency": "{:.0f} days",
            "Frequency": "{:.0f} orders"
        }),
        use_container_width=True
    )
    
    # 4. Model Performance charts
    st.markdown("### 📈 Churn Classifier Diagnostics")
    
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        # ROC Curve
        fpr, tpr, _ = roc_curve(df_churn["IsChurned_GroundTruth"], df_churn["ChurnProbability"])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.4f})", line=dict(color="#EF4444", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="grey")))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with diag_col2:
        # Confusion Matrix
        y_true = df_churn["IsChurned_GroundTruth"]
        y_pred = df_churn["PredictedChurn"]
        cm = confusion_matrix(y_true, y_pred)
        
        # Heatmap
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted Label", y="True Label"),
            x=["Active", "Churned"],
            y=["Active", "Churned"],
            color_continuous_scale="Reds",
            title="Confusion Matrix"
        )
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)
