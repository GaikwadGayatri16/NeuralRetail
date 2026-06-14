import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Inventory Optimization - NeuralRetail AI", page_icon="📦", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .status-alert {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📦 Inventory Optimization")
st.markdown("Group 3 Demand Forecasting & Inventory Dashboard. Optimize safety stock, ROP, and reordering.")

BASE_DIR = Path(__file__).resolve().parents[1]
abc_path = BASE_DIR / "outputs" / "abc_analysis.csv"
reorder_path = BASE_DIR / "outputs" / "reorder_recommendations.csv"

if not abc_path.exists() or not reorder_path.exists():
    st.warning("⚠️ Inventory optimization outputs not found. Please run the Inventory Optimization pipeline (`src/inventory.py`) to generate this data.")
else:
    @st.cache_data
    def load_inventory_data(a_path, r_path):
        abc_df = pd.read_csv(a_path)
        reorder_df = pd.read_csv(r_path)
        return abc_df, reorder_df
        
    df_abc, df_reorder = load_inventory_data(abc_path, reorder_path)
    
    # 1. ABC Analysis Summary
    st.markdown("### 📊 ABC Inventory Classification")
    st.write("Products are categorized based on their revenue contribution: **Class A** (Top 80% revenue, high value), **Class B** (Next 15%, medium value), **Class C** (Bottom 5%, low value).")
    
    col_abc1, col_abc2 = st.columns([1, 1])
    
    with col_abc1:
        # Group by ABC class
        class_summary = df_abc.groupby("ABC_Class").agg(
            ProductCount=("StockCode", "count"),
            TotalRevenue=("TotalRevenue", "sum")
        ).reset_index()
        
        # Donut Chart for Revenue Contribution
        fig_rev = px.pie(
            class_summary, names="ABC_Class", values="TotalRevenue",
            hole=0.4,
            title="Revenue Contribution by ABC Class",
            color="ABC_Class",
            color_discrete_map={"A": "#EC4899", "B": "#8B5CF6", "C": "#3B82F6"}
        )
        fig_rev.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rev, use_container_width=True)
        
    with col_abc2:
        # Donut Chart for Product Count Share
        fig_cnt = px.pie(
            class_summary, names="ABC_Class", values="ProductCount",
            hole=0.4,
            title="Product SKU Share by ABC Class",
            color="ABC_Class",
            color_discrete_map={"A": "#EC4899", "B": "#8B5CF6", "C": "#3B82F6"}
        )
        fig_cnt.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cnt, use_container_width=True)
        
    # 2. Reorder Recommendations
    st.markdown("### 🚨 Urgent Reorder Recommendations")
    
    recommended_items = df_reorder[df_reorder["ReorderRecommended"] == 1].copy()
    
    st.markdown(
        f"""
        <div class="status-alert">
            <h4>⚠️ Restock Needed: {len(recommended_items):,} SKUs require reordering.</h4>
            <p style="margin:0;">Current inventory levels for these items are equal to or lower than their calculated <b>Reorder Point (ROP)</b>. Quantities suggest the **Economic Order Quantity (EOQ)**.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Filter table options
    filter_abc = st.multiselect("Filter by ABC Class", options=["A", "B", "C"], default=["A", "B"])
    
    disp_reorder = recommended_items[recommended_items["ABC_Class"].isin(filter_abc)].sort_values(by="AvgDailyQty", ascending=False)
    
    st.dataframe(
        disp_reorder[[
            "StockCode", "Description", "ABC_Class", "Velocity", "AvgDailyQty",
            "SafetyStock", "ReorderPoint", "CurrentStock", "EOQ"
        ]].style.format({
            "AvgDailyQty": "{:.2f} units/day",
            "SafetyStock": "{:.0f} units",
            "ReorderPoint": "{:.0f} units",
            "CurrentStock": "{:.0f} units",
            "EOQ": "{:.0f} units"
        }),
        use_container_width=True
    )
    
    # 3. Product Velocity
    st.markdown("### 🏃 Product Velocity Analysis")
    
    col_vel1, col_vel2 = st.columns(2)
    
    with col_vel1:
        st.write("#### 🔥 Top Fast-Moving SKUs")
        fast_moving = df_reorder[df_reorder["Velocity"] == "Fast Moving"].sort_values(by="AvgDailyQty", ascending=False).head(20)
        st.dataframe(
            fast_moving[["StockCode", "Description", "ABC_Class", "AvgDailyQty"]].style.format({"AvgDailyQty": "{:.2f} units/day"}),
            use_container_width=True
        )
        
    with col_vel2:
        st.write("#### ❄️ Top Slow-Moving SKUs")
        slow_moving = df_reorder[df_reorder["Velocity"] == "Slow Moving"].sort_values(by="AvgDailyQty", ascending=True).head(20)
        st.dataframe(
            slow_moving[["StockCode", "Description", "ABC_Class", "AvgDailyQty"]].style.format({"AvgDailyQty": "{:.2f} units/day"}),
            use_container_width=True
        )
