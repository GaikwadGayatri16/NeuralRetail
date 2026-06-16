import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Inventory Optimization - NeuralRetail AI", page_icon="📦", layout="wide")

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
    
    .status-alert {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
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
        <h1 class="hero-title">📦 Inventory Optimization</h1>
        <p class="hero-desc">Run ABC inventory categorization, analyze daily velocities, and automate replenishment triggers (ROP, Safety Stock, EOQ).</p>
        <span class="hero-purpose">🎯 Business Purpose: Optimize warehousing cost structures, prevent costly stockouts of Class A items, and automate ordering formulas.</span>
    </div>
    """,
    unsafe_allow_html=True
)

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
    
    # Calculate pre-requisite KPI numbers
    total_skus = len(df_reorder)
    recommended_items = df_reorder[df_reorder["ReorderRecommended"] == 1]
    reorder_count = len(recommended_items)
    
    class_a_count = len(df_reorder[df_reorder["ABC_Class"] == "A"])
    fast_moving_count = len(df_reorder[df_reorder["Velocity"] == "Fast Moving"])
    
    # Draw KPI cards
    col1, col2, col3, col4 = st.columns(4)
    draw_kpi_card(col1, "SKUs Evaluated", f"{total_skus:,}", "📊")
    draw_kpi_card(col2, "Urgent Restock Triggers", f"{reorder_count:,}", "🚨")
    draw_kpi_card(col3, "Class A SKUs (Core Value)", f"{class_a_count:,}", "💎")
    draw_kpi_card(col4, "Fast-Moving Catalog Items", f"{fast_moving_count:,}", "🔥")
    
    st.divider()
    
    # Create Tabs
    tab_abc, tab_ranking, tab_safety, tab_rop, tab_recomm = st.tabs([
        "📊 ABC Analysis",
        "🏃 Inventory Ranking",
        "🛡️ Safety Stock Details",
        "🚨 Reorder Points Comparison",
        "📋 Replenishment Orders"
    ])
    
    # 1. ABC ANALYSIS TAB
    with tab_abc:
        st.subheader("ABC Classification Share breakdown")
        st.write("Products are categorized based on their revenue contribution: **Class A** (Top 80% revenue, high value), **Class B** (Next 15%, medium value), **Class C** (Bottom 5%, low value).")
        
        class_summary = df_abc.groupby("ABC_Class").agg(
            ProductCount=("StockCode", "count"),
            TotalRevenue=("TotalRevenue", "sum")
        ).reset_index()
        
        abc_chart_type = st.radio(
            "Select ABC Metric Chart",
            ["Revenue Contribution by ABC Class", "Product SKU Count Share by ABC Class"],
            horizontal=True
        )
        
        if abc_chart_type == "Revenue Contribution by ABC Class":
            fig_rev = px.pie(
                class_summary, names="ABC_Class", values="TotalRevenue",
                hole=0.4,
                title="Revenue Contribution by ABC Class",
                color="ABC_Class",
                color_discrete_map={"A": "#EC4899", "B": "#8B5CF6", "C": "#3B82F6"}
            )
            fig_rev.update_layout(
                template="plotly_dark", 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_rev, use_container_width=True)
            
            # Dynamic Insight
            a_rev = class_summary[class_summary["ABC_Class"] == "A"]["TotalRevenue"].values[0]
            total_rev = class_summary["TotalRevenue"].sum()
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: Revenue Contribution</div>
                    <p style="margin:0; color:#E2E8F0;">
                        <b>Class A</b> items represent <b>{(a_rev/total_rev * 100):.2f}%</b> of total store revenue 
                        (valued at <b>${a_rev:,.2f}</b>). These high-value SKUs demand close daily monitoring.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            fig_cnt = px.pie(
                class_summary, names="ABC_Class", values="ProductCount",
                hole=0.4,
                title="Product SKU Share by ABC Class",
                color="ABC_Class",
                color_discrete_map={"A": "#EC4899", "B": "#8B5CF6", "C": "#3B82F6"}
            )
            fig_cnt.update_layout(
                template="plotly_dark", 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_cnt, use_container_width=True)
            
            # Dynamic Insight
            a_count = class_summary[class_summary["ABC_Class"] == "A"]["ProductCount"].values[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: SKU Share Ratio</div>
                    <p style="margin:0; color:#E2E8F0;">
                        <b>Class A</b> makes up only <b>{(a_count/total_skus * 100):.2f}%</b> of total catalog items ({a_count:,} SKUs), 
                        yet produces the overwhelming majority of revenue, illustrating the classic Pareto Principle.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # 2. INVENTORY RANKING TAB
    with tab_ranking:
        st.subheader("Inventory Stock Velocity Categorization")
        
        velocity_cat = st.radio(
            "Select Velocity Category",
            ["🔥 Top Fast-Moving SKUs", "❄️ Top Slow-Moving SKUs"],
            horizontal=True
        )
        
        if velocity_cat == "🔥 Top Fast-Moving SKUs":
            fast_moving = df_reorder[df_reorder["Velocity"] == "Fast Moving"].sort_values(by="AvgDailyQty", ascending=False).head(50)
            st.markdown("These products exhibit high daily sales quantities. Maintaining adequate stock is crucial to avoid lost revenue.")
            st.dataframe(
                fast_moving[["StockCode", "Description", "ABC_Class", "AvgDailyQty"]].style.format({"AvgDailyQty": "{:.2f} units/day"}),
                use_container_width=True
            )
            
            # Dynamic Insight
            fastest_item = fast_moving.iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: Fast-Moving Catalog Performance</div>
                    <p style="margin:0; color:#E2E8F0;">
                        The fastest-moving item is <b>{fastest_item['Description']}</b>, selling an average of 
                        <b>{fastest_item['AvgDailyQty']:.2f} units</b> per day.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            slow_moving = df_reorder[df_reorder["Velocity"] == "Slow Moving"].sort_values(by="AvgDailyQty", ascending=True).head(50)
            st.markdown("These products have minimal daily inventory movements. Review these items for potential clearance or storage reduction.")
            st.dataframe(
                slow_moving[["StockCode", "Description", "ABC_Class", "AvgDailyQty"]].style.format({"AvgDailyQty": "{:.2f} units/day"}),
                use_container_width=True
            )
            
            # Dynamic Insight
            slowest_item = slow_moving.iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: Slow-Moving Stock Assessment</div>
                    <p style="margin:0; color:#E2E8F0;">
                        The slowest-moving active item is <b>{slowest_item['Description']}</b>, averaging only 
                        <b>{slowest_item['AvgDailyQty']:.4f} units</b> sold per day.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # 3. SAFETY STOCK TAB
    with tab_safety:
        st.subheader("Safety Stock Distribution Levels")
        
        fig_safety = px.scatter(
            df_reorder, x="AvgDailyQty", y="SafetyStock", color="ABC_Class",
            title="Safety Stock Levels vs. Average Daily Demand",
            labels={"AvgDailyQty": "Average Daily Quantity Demand", "SafetyStock": "Calculated Safety Stock (Units)"},
            hover_name="Description",
            color_discrete_map={"A": "#EC4899", "B": "#8B5CF6", "C": "#3B82F6"}
        )
        fig_safety.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_safety, use_container_width=True)
        
        # Dynamic Insight
        avg_safety = df_reorder["SafetyStock"].mean()
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Safety Buffer Valuation</div>
                <p style="margin:0; color:#E2E8F0;">
                    The average safety stock level across all evaluated products is <b>{avg_safety:.1f} units</b>. 
                    Safety stock provides a buffer against supply chain lead time volatility and random demand surges.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 4. REORDER POINTS TAB
    with tab_rop:
        st.subheader("Current Stock Levels vs. Reorder Points (ROP)")
        
        fig_rop = px.scatter(
            df_reorder, x="ReorderPoint", y="CurrentStock", color="ReorderRecommended",
            title="Comparison: Current Stock vs. Calculated Reorder Point (ROP)",
            labels={"ReorderPoint": "Reorder Point (Units)", "CurrentStock": "Current Inventory Level", "ReorderRecommended": "Reorder Flag"},
            hover_name="Description",
            color_discrete_map={0: "#10B981", 1: "#EF4444"}
        )
        # Add a y=x reference line
        fig_rop.add_shape(
            type="line", line=dict(dash="dash", color="grey"),
            x0=0, y0=0, x1=df_reorder["ReorderPoint"].max(), y1=df_reorder["ReorderPoint"].max()
        )
        fig_rop.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_rop, use_container_width=True)
        
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Reorder Triggers (ROP)</div>
                <p style="margin:0; color:#E2E8F0;">
                    Items located <b>below the dashed reference line</b> (highlighted in red) have current stocks less than or equal to their Reorder Point, 
                    signaling a prompt replenishment order to avoid stockout.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 5. RECOMMENDATIONS TAB
    with tab_recomm:
        st.subheader("Urgent Stock Replenishment Orders")
        
        st.markdown(
            f"""
            <div class="status-alert">
                <h4 style="margin:0; color: #FBBF24;">⚠️ Restock Action Needed: {reorder_count:,} SKUs require reordering.</h4>
                <p style="margin-top: 0.5rem; margin-bottom: 0; color: #E2E8F0;">Current inventory levels for these items are equal to or lower than their calculated <b>Reorder Point (ROP)</b>. Quantities suggest the **Economic Order Quantity (EOQ)**.</p>
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
        
        # Dynamic Insight
        class_a_reorder_count = len(disp_reorder[disp_reorder["ABC_Class"] == "A"])
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Key Insight: Restocking Priorities</div>
                <p style="margin:0; color:#E2E8F0;">
                    Within the filtered set, there are <b>{class_a_reorder_count} Class A</b> items flagged for urgent reordering. 
                    These high-value items must be restocked first to protect major store revenue streams.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

