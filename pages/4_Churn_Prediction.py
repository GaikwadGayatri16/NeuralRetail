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
    
    .search-panel {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }
    
    .metric-panel {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
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
    
    /* Insight Card Styles */
    .insight-card {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .insight-title {
        color: #F87171;
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
        <h1 class="hero-title">⚠️ Churn Prediction</h1>
        <p class="hero-desc">Identify high-risk shoppers, diagnose churn probability distributions, and interpret model features.</p>
        <span class="hero-purpose">🎯 Business Purpose: Mitigate customer churn and retain value through early detection of declining transaction frequency.</span>
    </div>
    """,
    unsafe_allow_html=True
)

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
    
    df_churn["Customer ID"] = df_churn["Customer ID"].astype(str)
    
    # Precalculate metrics for header KPI cards
    avg_churn_prob = df_churn["ChurnProbability"].mean()
    high_risk_count = len(df_churn[df_churn["ChurnProbability"] >= 0.70])
    ground_truth_churned = len(df_churn[df_churn["IsChurned_GroundTruth"] == 1])
    
    # Calculate model AUC dynamically for the metric card
    fpr, tpr, _ = roc_curve(df_churn["IsChurned_GroundTruth"], df_churn["ChurnProbability"])
    roc_auc = auc(fpr, tpr)
    
    # Display top-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    draw_kpi_card(col1, "Average Churn Probability", f"{avg_churn_prob:.1%}", "📊")
    draw_kpi_card(col2, "High-Risk Accounts (Prob ≥ 70%)", f"{high_risk_count:,}", "🚨")
    draw_kpi_card(col3, "Ground-Truth Churned", f"{ground_truth_churned:,}", "📉")
    draw_kpi_card(col4, "Model AUC Score", f"{roc_auc:.4f}", "🏆")
    
    st.divider()
    
    # Create Tabs
    tab_dist, tab_risk, tab_feat, tab_roc, tab_perf = st.tabs([
        "📊 Churn Distribution",
        "🎯 High Risk Customers",
        "🔑 Feature Importance",
        "📈 ROC Curve",
        "⚙️ Model Performance"
    ])
    
    # 1. CHURN DISTRIBUTION
    with tab_dist:
        st.subheader("Distribution of Customer Churn Probabilities")
        fig_dist = px.histogram(
            df_churn, x="ChurnProbability", nbins=50,
            title="Customer Count by Churn Risk Probability",
            labels={"ChurnProbability": "Churn Probability Level"},
            color_discrete_sequence=["#EF4444"]
        )
        fig_dist.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Dynamic Insight
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Churn Risk Distribution</div>
                <p style="margin:0; color:#E2E8F0;">
                    A total of <b>{high_risk_count:,}</b> customers have exceeded a <b>70%</b> probability of churn. 
                    The histogram displays a bimodal pattern typical of datasets split between highly active shoppers and dormant profiles.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 2. HIGH RISK CUSTOMERS
    with tab_risk:
        st.subheader("Target Marketing Campaign list & Risk Lookup")
        
        # Risk Lookup Search
        st.markdown('<div class="search-panel">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Customer Risk Profile Assessment")
        search_id = st.text_input("Enter Customer ID:", placeholder="e.g. 17850")
        
        if search_id:
            cust_row = df_churn[df_churn["Customer ID"] == search_id.strip()]
            if cust_row.empty:
                st.error(f"Customer ID `{search_id}` not found.")
            else:
                row = cust_row.iloc[0]
                prob = row["ChurnProbability"]
                
                if prob >= 0.70:
                    risk_badge = '<span class="high-risk-badge">High Risk (🚨)</span>'
                elif prob >= 0.30:
                    risk_badge = '<span class="medium-risk-badge">Medium Risk (⚠️)</span>'
                else:
                    risk_badge = '<span class="low-risk-badge">Low Risk (✅)</span>'
                    
                st.markdown(f"**Customer Status:** {risk_badge}", unsafe_allow_html=True)
                
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                col_c1.markdown(f'<div class="metric-panel"><b>Recency</b><br>{int(row["Recency"])} days</div>', unsafe_allow_html=True)
                col_c2.markdown(f'<div class="metric-panel"><b>Frequency</b><br>{int(row["Frequency"])} orders</div>', unsafe_allow_html=True)
                col_c3.markdown(f'<div class="metric-panel"><b>Monetary</b><br>${row["Monetary"]:,.2f}</div>', unsafe_allow_html=True)
                col_c4.markdown(f'<div class="metric-panel"><b>LTV</b><br>${row["LifetimeRevenue"]:,.2f}</div>', unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div style="margin-top: 1rem;">
                        • <b>Shopper Age:</b> {int(row['CustomerAgeDays'])} days<br>
                        • <b>Days Since Last Purchase:</b> {int(row['DaysSinceLastPurchase'])} days<br>
                        • <b>Average Order Value (AOV):</b> ${row['AverageOrderValue']:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # High Risk Target Group
        st.markdown("#### 📋 High Churn Risk Target Group (Top 100 Active-at-Risk)")
        
        high_risk_list = df_churn[
            (df_churn["ChurnProbability"] >= 0.70) & 
            (df_churn["DaysSinceLastPurchase"] <= 90)
        ].sort_values(by="ChurnProbability", ascending=False).head(100)
        
        if high_risk_list.empty:
            high_risk_list = df_churn[df_churn["DaysSinceLastPurchase"] <= 90].sort_values(by="ChurnProbability", ascending=False).head(100)
            
        st.markdown("These customers have purchased within the last 90 days, but show high probabilities of churning based on their transaction patterns (dropping frequency, lowering spend). **Target them immediately with win-back campaigns!**")
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
        
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Target Marketing Campaigns</div>
                <p style="margin:0; color:#E2E8F0;">
                    Targeting this subgroup preserves customer acquisition costs. A coupon or win-back offer sent to these <b>{len(high_risk_list)}</b> customers has a high potential rate of return before they cross into permanent churn.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 3. FEATURE IMPORTANCE
    with tab_feat:
        st.subheader("Model Feature Weights & Drivers")
        
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
                title=f"Feature Weights ({model_dict['model_name']})",
                color="Importance", color_continuous_scale="Reds"
            )
            fig_feat.update_layout(
                template="plotly_dark", 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            
            # Dynamic Insight
            top_feat_name = feat_df.sort_values(by="Importance", ascending=False).iloc[0]["Feature"]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: Churn Drivers</div>
                    <p style="margin:0; color:#E2E8F0;">
                        The most significant predictor of customer retention behavior is <b>{top_feat_name}</b>. 
                        Changes in this specific behavioral signature heavily influence the model's output scores.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Feature importance metrics not available for this classifier architecture.")
            
    # 4. ROC CURVE
    with tab_roc:
        st.subheader("Receiver Operating Characteristic (ROC) Diagnostic")
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.4f})", line=dict(color="#EF4444", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="grey")))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: ROC Accuracy Diagnostic</div>
                <p style="margin:0; color:#E2E8F0;">
                    An Area Under the Curve (AUC) of <b>{roc_auc:.4f}</b> shows highly robust model performance, indicating that the model 
                    retains a high capability to differentiate between active and churning customer cohorts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 5. MODEL PERFORMANCE
    with tab_perf:
        st.subheader("Confusion Matrix & Binary Classification Stats")
        
        y_true = df_churn["IsChurned_GroundTruth"]
        y_pred = df_churn["PredictedChurn"]
        cm = confusion_matrix(y_true, y_pred)
        
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted Label", y="True Label"),
            x=["Active", "Churned"],
            y=["Active", "Churned"],
            color_continuous_scale="Reds",
            title="Confusion Matrix Heatmap"
        )
        fig_cm.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Calculate rates
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Precision", f"{precision:.2%}", "TP / (TP + FP)")
        col_m2.metric("Recall (Sensitivity)", f"{recall:.2%}", "TP / (TP + FN)")
        col_m3.metric("Overall Accuracy", f"{accuracy:.2%}", "(TP + TN) / Total")
        
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Model Diagnostics Summary</div>
                <p style="margin:0; color:#E2E8F0;">
                    Out of <b>{(tp+fn):,}</b> actual churned accounts, the classifier successfully identified <b>{tp:,}</b> profiles (Recall of <b>{recall:.2%}</b>), 
                    ensuring that promotional outreach captures the maximum percentage of real risk cases.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

