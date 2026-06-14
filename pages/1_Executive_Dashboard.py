import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Executive Dashboard - NeuralRetail AI", page_icon="📊", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .kpi-title {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Executive Dashboard")
st.markdown("High-level key performance indicators and revenue trends.")

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
    st.sidebar.header("Filter Analytics")
    
    # 1. Date filter
    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()
    
    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    # 2. Country filter
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
        
    # Check if empty
    if filtered_df.empty:
        st.warning("No transactions match the selected filters.")
    else:
        # Calculate KPIs
        total_revenue = filtered_df["TotalPrice"].sum()
        total_orders = filtered_df["Invoice"].nunique()
        total_customers = filtered_df["Customer ID"].nunique()
        total_products = filtered_df["StockCode"].nunique()
        
        # Display KPI Cards
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-title">Total Revenue</div>
                    <div class="kpi-value">${total_revenue:,.2f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Total Orders</div>
                    <div class="kpi-value">{total_orders:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Active Customers</div>
                    <div class="kpi-value">{total_customers:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Products Sold</div>
                    <div class="kpi-value">{total_products:,}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Charts section
        st.markdown("### 📈 Revenue Trends")
        
        # Aggregate monthly revenue
        monthly_df = filtered_df.copy()
        monthly_df["MonthYear"] = monthly_df["InvoiceDate"].dt.to_period("M").astype(str)
        monthly_rev = monthly_df.groupby("MonthYear")["TotalPrice"].sum().reset_index()
        
        # Aggregate daily revenue
        daily_df = filtered_df.copy()
        daily_df["Date"] = daily_df["InvoiceDate"].dt.date
        daily_rev = daily_df.groupby("Date")["TotalPrice"].sum().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_daily = px.line(
                daily_rev, x="Date", y="TotalPrice",
                title="Daily Revenue Trend",
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
            
        with col2:
            fig_monthly = px.bar(
                monthly_rev, x="MonthYear", y="TotalPrice",
                title="Monthly Revenue Trend",
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
            
        # Summary distribution info
        st.markdown("### 🗺️ Revenue by Country Breakdown")
        country_rev = filtered_df.groupby("Country")["TotalPrice"].sum().reset_index()
        country_rev = country_rev.sort_values(by="TotalPrice", ascending=False).head(10)
        
        fig_country = px.bar(
            country_rev, x="TotalPrice", y="Country", orientation="h",
            title="Top 10 Countries by Revenue",
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
