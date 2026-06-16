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
    }
    
    .insight-card {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Sales Analytics")
st.markdown("Analyze sales performance, top catalog items, and order distributions over time and geography.")
st.divider()

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
        col1.metric("Revenue Volume", f"${total_revenue:,.2f}")
        col2.metric("Total Units Sold", f"{total_units:,}")
        col3.metric("Average Transaction Value", f"${avg_order_val:,.2f}")
        col4.metric("Top SKU (by Quantity)", f"{pop_prod[:22]}...")
        
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
                yaxis=dict(autorange="reversed"), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_top_rev, use_container_width=True)
            
            # Dynamic Insight
            top_rev_sku = top_rev_products.iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">💡 Top Revenue Product Insight</div>
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
                yaxis=dict(autorange="reversed"), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_top_qty, use_container_width=True)
            
            # Dynamic Insight
            top_qty_sku = top_qty_products.iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">💡 Physical Volume Insight</div>
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
            fig_geo.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_geo, use_container_width=True)
            
            # Dynamic Insight
            top_country = country_shares.iloc[0]
            top_country_percentage = (top_country['TotalPrice'] / total_revenue * 100) if total_revenue > 0 else 0
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">💡 Geographical Insight</div>
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
            fig_weekday.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_weekday, use_container_width=True)
            
            # Dynamic Insight
            if not weekday_sales.empty:
                best_day_row = weekday_sales.loc[weekday_sales["TotalPrice"].idxmax()]
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-title">💡 Weekly Activity Insight</div>
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
                xaxis_showgrid=False
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Dynamic Insight
            if not monthly_sales.empty:
                best_month = monthly_sales.loc[monthly_sales["TotalPrice"].idxmax()]
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-title">💡 Monthly Trend Insight</div>
                        <p style="margin:0; color:#E2E8F0;">
                            Monthly revenue reached its peak in <b>{best_month['MonthYear']}</b> with a valuation of 
                            <b>${best_month['TotalPrice']:,.2f}</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
