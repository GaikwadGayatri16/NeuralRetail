import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Sales Analytics - NeuralRetail AI", page_icon="📈", layout="wide")

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

    /* Insight Card Styles */
    .insight-card {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .insight-title {
        color: #34D399;
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
        <h1 class="hero-title">📈 Sales Analytics</h1>
        <p class="hero-desc">Analyze sales performance, top catalog items, and order distributions over time and geography.</p>
        <span class="hero-purpose">🎯 Business Purpose: Identify product sales trends, top-revenue items, physical sales volume, and sales cycles.</span>
    </div>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parents[1]
data_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"

if not data_path.exists():
    st.error(f"Cleaned dataset not found at `{data_path}`. Please run the preprocessing pipeline first.")
else:
    @st.cache_data
    def load_data(path):
        df = pd.read_csv(path)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        return df
        
    df = load_data(data_path)
    
    # Date Range Selector
    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()
    
    st.sidebar.header("⚙️ Date Settings")
    start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    filtered_df = df[
        (df["InvoiceDate"].dt.date >= start_date) & 
        (df["InvoiceDate"].dt.date <= end_date)
    ]
    
    if filtered_df.empty:
        st.warning("⚠️ No sales transactions found for this date range. Adjust the dates in the sidebar.")
    else:
        # Calculate premium KPIs for the header
        total_revenue = filtered_df["TotalPrice"].sum()
        total_units = filtered_df["Quantity"].sum()
        total_orders = filtered_df["Invoice"].nunique()
        avg_order_val = total_revenue / total_orders if total_orders > 0 else 0
        
        # Most popular product in selection
        pop_prod_row = filtered_df.groupby("Description")["Quantity"].sum().reset_index()
        pop_prod = pop_prod_row.sort_values(by="Quantity", ascending=False).iloc[0]["Description"] if not pop_prod_row.empty else "N/A"
        
        # Display KPI cards
        col1, col2, col3, col4 = st.columns(4)
        draw_kpi_card(col1, "Revenue Volume", f"${total_revenue:,.2f}", "💰")
        draw_kpi_card(col2, "Total Units Sold", f"{total_units:,}", "📦")
        draw_kpi_card(col3, "Average Order Value", f"${avg_order_val:,.2f}", "📊")
        draw_kpi_card(col4, "Top SKU (by Quantity)", f"{pop_prod[:20]}...", "🏆")
        
        st.divider()
        
        # Create Tabs
        tab_rev_prod, tab_qty_prod, tab_geo, tab_weekly, tab_monthly = st.tabs([
            "🏆 Top Revenue Products",
            "📦 Top Quantity Products",
            "🌍 Country Revenue",
            "📅 Weekly Pattern",
            "📈 Monthly Trend"
        ])
        
        # 1. TOP REVENUE PRODUCTS
        with tab_rev_prod:
            st.subheader("Top 10 Products by Total Revenue Contribution")
            top_rev_products = filtered_df.groupby("Description")["TotalPrice"].sum().reset_index()
            top_rev_products = top_rev_products.sort_values(by="TotalPrice", ascending=False).head(10)
            
            fig_top_rev = px.bar(
                top_rev_products, x="TotalPrice", y="Description", orientation="h",
                title="Top 10 Products by Total Revenue",
                labels={"TotalPrice": "Revenue ($)", "Description": "Product Description"},
                color="TotalPrice", color_continuous_scale="Viridis"
            )
            fig_top_rev.update_layout(
                template="plotly_dark", 
                yaxis=dict(autorange="reversed", showgrid=True, gridcolor="rgba(255,255,255,0.05)"), 
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_top_rev, use_container_width=True)
            
            # Dynamic Insight
            top_rev_sku = top_rev_products.iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: Top Revenue Driver</div>
                    <p style="margin:0; color:#E2E8F0;">
                        The product generating the highest value is <b>{top_rev_sku['Description']}</b>, bringing in 
                        <b>${top_rev_sku['TotalPrice']:,.2f}</b> in sales during this timeframe.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # 2. TOP QUANTITY PRODUCTS
        with tab_qty_prod:
            st.subheader("Top 10 Products by Physical Sales Volume")
            top_qty_products = filtered_df.groupby("Description")["Quantity"].sum().reset_index()
            top_qty_products = top_qty_products.sort_values(by="Quantity", ascending=False).head(10)
            
            fig_top_qty = px.bar(
                top_qty_products, x="Quantity", y="Description", orientation="h",
                title="Top 10 Products by Quantity Sold",
                labels={"Quantity": "Units Sold", "Description": "Product Description"},
                color="Quantity", color_continuous_scale="Cividis"
            )
            fig_top_qty.update_layout(
                template="plotly_dark", 
                yaxis=dict(autorange="reversed", showgrid=True, gridcolor="rgba(255,255,255,0.05)"), 
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_top_qty, use_container_width=True)
            
            # Dynamic Insight
            top_qty_sku = top_qty_products.iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: Catalog Leader by Volume</div>
                    <p style="margin:0; color:#E2E8F0;">
                        The catalog item with the highest unit volume is <b>{top_qty_sku['Description']}</b> with 
                        <b>{top_qty_sku['Quantity']:,}</b> physical units distributed.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # 3. COUNTRY REVENUE
        with tab_geo:
            st.subheader("Geographical Distribution and Revenue Shares")
            country_shares = filtered_df.groupby("Country")["TotalPrice"].sum().reset_index()
            country_shares = country_shares.sort_values(by="TotalPrice", ascending=False)
            
            fig_geo = px.treemap(
                country_shares, path=["Country"], values="TotalPrice",
                title="Revenue Contribution by Country",
                color="TotalPrice", color_continuous_scale="RdBu"
            )
            fig_geo.update_layout(
                template="plotly_dark", 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_geo, use_container_width=True)
            
            # Dynamic Insight
            top_country = country_shares.iloc[0]
            top_country_percentage = (top_country['TotalPrice'] / total_revenue * 100) if total_revenue > 0 else 0
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">📌 Key Insight: Top Performing Region</div>
                    <p style="margin:0; color:#E2E8F0;">
                        <b>{top_country['Country']}</b> is the dominant market region, representing <b>{top_country_percentage:.2f}%</b> of total filtered transactions with a valuation of <b>${top_country['TotalPrice']:,.2f}</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # 4. WEEKLY PATTERN
        with tab_weekly:
            st.subheader("Weekly Activity Profile (Revenue by Day of Week)")
            filtered_df["Weekday"] = filtered_df["InvoiceDate"].dt.day_name()
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # Reindex to ensure chronological sequence
            weekday_sales = filtered_df.groupby("Weekday")["TotalPrice"].sum().reindex(weekday_order).dropna().reset_index()
            
            fig_weekday = px.bar(
                weekday_sales, x="Weekday", y="TotalPrice",
                title="Weekly Sales Pattern",
                labels={"TotalPrice": "Revenue ($)", "Weekday": "Day of the Week"},
                color="TotalPrice", color_continuous_scale="Teal"
            )
            fig_weekday.update_layout(
                template="plotly_dark", 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_weekday, use_container_width=True)
            
            # Dynamic Insight
            if not weekday_sales.empty:
                best_day_row = weekday_sales.loc[weekday_sales["TotalPrice"].idxmax()]
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-title">📌 Key Insight: Optimal Sales Day</div>
                        <p style="margin:0; color:#E2E8F0;">
                            <b>{best_day_row['Weekday']}</b> is the highest performing weekday, contributing 
                            <b>${best_day_row['TotalPrice']:,.2f}</b> in sales volume.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # 5. MONTHLY TREND
        with tab_monthly:
            st.subheader("Monthly Revenue Trends and Cycles")
            
            monthly_sales_df = filtered_df.copy()
            monthly_sales_df["MonthYear"] = monthly_sales_df["InvoiceDate"].dt.to_period("M").astype(str)
            monthly_sales = monthly_sales_df.groupby("MonthYear")["TotalPrice"].sum().reset_index()
            
            fig_monthly = px.line(
                monthly_sales, x="MonthYear", y="TotalPrice",
                title="Monthly Cumulative Sales Revenue ($)",
                labels={"TotalPrice": "Revenue ($)", "MonthYear": "Month"},
                markers=True,
                color_discrete_sequence=["#F59E0B"]
            )
            fig_monthly.update_layout(
                template="plotly_dark", 
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Dynamic Insight
            if not monthly_sales.empty:
                best_month = monthly_sales.loc[monthly_sales["TotalPrice"].idxmax()]
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-title">📌 Key Insight: Monthly Revenue Peak</div>
                        <p style="margin:0; color:#E2E8F0;">
                            Monthly revenue reached its peak in <b>{best_month['MonthYear']}</b> with a valuation of 
                            <b>${best_month['TotalPrice']:,.2f}</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

