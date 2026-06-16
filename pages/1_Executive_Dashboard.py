import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Executive Dashboard - NeuralRetail AI", page_icon="📊", layout="wide")

# Custom premium CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .insight-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Executive Dashboard")
st.markdown("Analyze executive-level key performance indicators, revenue distributions, and high-level trends.")
st.divider()

# Load Data
BASE_DIR = Path(__file__).resolve().parents[1]
data_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"

if not data_path.exists():
    st.error(f"Cleaned dataset not found at `{data_path}`. Please run the data preprocessing pipeline first.")
else:
    @st.cache_data
    def load_data(path):
        df = pd.read_csv(path)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        return df
        
    df = load_data(data_path)
    
    # Setup Filters
    st.sidebar.header("⚙️ Filter Panel")
    
    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()
    
    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    countries = sorted(df["Country"].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries (Default: All)",
        options=countries,
        default=[]
    )
    
    # Apply filters
    filtered_df = df[
        (df["InvoiceDate"].dt.date >= start_date) & 
        (df["InvoiceDate"].dt.date <= end_date)
    ]
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]
        
    if filtered_df.empty:
        st.warning("⚠️ No transactions match the selected filters. Please adjust the sidebar filters.")
    else:
        # Calculate KPIs
        total_revenue = filtered_df["TotalPrice"].sum()
        total_orders = filtered_df["Invoice"].nunique()
        total_customers = filtered_df["Customer ID"].nunique()
        total_products = filtered_df["StockCode"].nunique()
        
        # Display KPI Cards before charts
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")
        col2.metric(label="Total Orders", value=f"{total_orders:,}")
        col3.metric(label="Active Customers", value=f"{total_customers:,}")
        col4.metric(label="Unique Products", value=f"{total_products:,}")
        
        st.divider()
        
        # Create Tabs
        tab_overview, tab_revenue, tab_customer, tab_country = st.tabs([
            "📋 Overview", 
            "📈 Revenue Trends", 
            "👥 Customer Metrics", 
            "🗺️ Country Performance"
        ])
        
        # 1. OVERVIEW TAB
        with tab_overview:
            st.subheader("Dashboard Overview & Dataset Scope")
            st.markdown(
                f"""
                This dashboard presents aggregated transactional insights filtered by country and date selection.
                - **Active Filter Period**: `{start_date}` to `{end_date}`
                - **Scope of Data**: `{len(filtered_df):,}` transactions analyzed.
                - **Average Transaction Value**: `${(total_revenue / total_orders if total_orders > 0 else 0):,.2f}`
                """
            )
            st.markdown("#### Preview of Cleaned Transactions")
            st.dataframe(filtered_df.head(10), use_container_width=True)
            
            st.markdown(
                """
                <div class="insight-card">
                    <div class="insight-title">💡 General Context</div>
                    <p style="margin:0; color:#E2E8F0;">
                        Use the date filter in the sidebar to restrict observations to a specific quarter, or filter by specific countries to assess local market sizes.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # 2. REVENUE TRENDS TAB
        with tab_revenue:
            st.subheader("Sales Revenue Analysis")
            
            # Sub-toggle inside tab to keep charts large
            granularity = st.radio(
                "Choose Revenue Trend Granularity", 
                ["Daily Revenue Trend", "Monthly Revenue Trend"],
                horizontal=True
            )
            
            if granularity == "Daily Revenue Trend":
                daily_df = filtered_df.copy()
                daily_df["Date"] = daily_df["InvoiceDate"].dt.date
                daily_rev = daily_df.groupby("Date")["TotalPrice"].sum().reset_index()
                
                fig_daily = px.line(
                    daily_rev, x="Date", y="TotalPrice",
                    title="Daily Sales Revenue ($) Trend",
                    labels={"TotalPrice": "Sales Revenue ($)", "Date": "Date"},
                    color_discrete_sequence=["#6366F1"]
                )
                fig_daily.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_showgrid=False
                )
                st.plotly_chart(fig_daily, use_container_width=True)
                
                # Dynamic Daily Insight
                max_day_row = daily_rev.loc[daily_rev["TotalPrice"].idxmax()]
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-title">💡 Revenue Trend Insight</div>
                        <p style="margin:0; color:#E2E8F0;">
                            Peak daily sales occurred on <b>{max_day_row['Date'].strftime('%d %B %Y')}</b> generating a total of 
                            <b>${max_day_row['TotalPrice']:,.2f}</b> in revenue.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            else:
                monthly_df = filtered_df.copy()
                monthly_df["MonthYear"] = monthly_df["InvoiceDate"].dt.to_period("M").astype(str)
                monthly_rev = monthly_df.groupby("MonthYear")["TotalPrice"].sum().reset_index()
                
                fig_monthly = px.bar(
                    monthly_rev, x="MonthYear", y="TotalPrice",
                    title="Monthly Sales Revenue ($) Trend",
                    labels={"TotalPrice": "Sales Revenue ($)", "MonthYear": "Month"},
                    color_discrete_sequence=["#10B981"]
                )
                fig_monthly.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_showgrid=False
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Dynamic Monthly Insight
                max_month_row = monthly_rev.loc[monthly_rev["TotalPrice"].idxmax()]
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-title">💡 Monthly Revenue Trend Insight</div>
                        <p style="margin:0; color:#E2E8F0;">
                            The highest performing month in this scope was <b>{max_month_row['MonthYear']}</b> with 
                            <b>${max_month_row['TotalPrice']:,.2f}</b> in sales.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # 3. CUSTOMER METRICS TAB
        with tab_customer:
            st.subheader("Top Customers & Spend Statistics")
            
            # Aggregate customer spend
            customer_spend = filtered_df.groupby("Customer ID")["TotalPrice"].sum().reset_index()
            customer_spend = customer_spend.sort_values(by="TotalPrice", ascending=False).head(10)
            customer_spend["Customer ID"] = customer_spend["Customer ID"].astype(str)
            
            fig_cust = px.bar(
                customer_spend, x="TotalPrice", y="Customer ID", orientation="h",
                title="Top 10 Customers by Revenue Contribution",
                labels={"TotalPrice": "Total Spend ($)", "Customer ID": "Customer ID"},
                color="TotalPrice",
                color_continuous_scale="Agsunset"
            )
            fig_cust.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_cust, use_container_width=True)
            
            # Dynamic Insight
            top_cust = customer_spend.iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">💡 Customer Spend Insight</div>
                    <p style="margin:0; color:#E2E8F0;">
                        The top customer is ID <b>{top_cust['Customer ID']}</b>, contributing <b>${top_cust['TotalPrice']:,.2f}</b> to the total sales volume in the selected date range.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # 4. COUNTRY PERFORMANCE TAB
        with tab_country:
            st.subheader("Geographical Distribution of Revenue")
            
            country_rev = filtered_df.groupby("Country")["TotalPrice"].sum().reset_index()
            country_rev = country_rev.sort_values(by="TotalPrice", ascending=False).head(10)
            
            fig_country = px.bar(
                country_rev, x="TotalPrice", y="Country", orientation="h",
                title="Top 10 Countries by Revenue Contribution",
                labels={"TotalPrice": "Revenue ($)", "Country": "Country"},
                color="TotalPrice",
                color_continuous_scale="Plasma"
            )
            fig_country.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_country, use_container_width=True)
            
            # Dynamic Country Performance Insight
            total_filtered_rev = filtered_df["TotalPrice"].sum()
            uk_rev = filtered_df[filtered_df["Country"] == "United Kingdom"]["TotalPrice"].sum()
            uk_percentage = (uk_rev / total_filtered_rev * 100) if total_filtered_rev > 0 else 0
            
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">💡 Geographical Insight</div>
                    <p style="margin:0; color:#E2E8F0;">
                        The <b>United Kingdom</b> represents the core of the business, accounting for <b>{uk_percentage:.2f}%</b> of total filtered sales revenue (${uk_rev:,.2f}).
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
