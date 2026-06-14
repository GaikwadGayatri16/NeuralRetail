import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Set page settings
st.set_page_config(page_title="Sales Analytics - NeuralRetail AI", page_icon="📈", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .panel {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Sales Analytics")
st.markdown("Detailed exploration of sales trends, geographical performance, and product distributions.")

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
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col_f2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
    filtered_df = df[
        (df["InvoiceDate"].dt.date >= start_date) & 
        (df["InvoiceDate"].dt.date <= end_date)
    ]
    
    if filtered_df.empty:
        st.warning("No sales transactions found for this date range.")
    else:
        # 1. Product Performance
        st.markdown("### 🏷️ Product Performance")
        
        prod_col1, prod_col2 = st.columns(2)
        
        with prod_col1:
            top_rev_products = filtered_df.groupby("Description")["TotalPrice"].sum().reset_index()
            top_rev_products = top_rev_products.sort_values(by="TotalPrice", ascending=False).head(10)
            
            fig_top_rev = px.bar(
                top_rev_products, x="TotalPrice", y="Description", orientation="h",
                title="Top 10 Products by Total Revenue",
                labels={"TotalPrice": "Revenue ($)", "Description": "Product Name"},
                color="TotalPrice", color_continuous_scale="Viridis"
            )
            fig_top_rev.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_top_rev, use_container_width=True)
            
        with prod_col2:
            top_qty_products = filtered_df.groupby("Description")["Quantity"].sum().reset_index()
            top_qty_products = top_qty_products.sort_values(by="Quantity", ascending=False).head(10)
            
            fig_top_qty = px.bar(
                top_qty_products, x="Quantity", y="Description", orientation="h",
                title="Top 10 Products by Quantity Sold",
                labels={"Quantity": "Units Sold", "Description": "Product Name"},
                color="Quantity", color_continuous_scale="Cividis"
            )
            fig_top_qty.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_top_qty, use_container_width=True)
            
        # 2. Geographical and Weekday Analysis
        st.markdown("### 🌍 Market & Schedule Analysis")
        
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            country_shares = filtered_df.groupby("Country")["TotalPrice"].sum().reset_index()
            country_shares = country_shares.sort_values(by="TotalPrice", ascending=False)
            
            fig_geo = px.treemap(
                country_shares, path=["Country"], values="TotalPrice",
                title="Revenue Contribution by Country",
                color="TotalPrice", color_continuous_scale="RdBu"
            )
            fig_geo.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with m_col2:
            # Weekday sales analysis
            filtered_df["Weekday"] = filtered_df["InvoiceDate"].dt.day_name()
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_sales = filtered_df.groupby("Weekday")["TotalPrice"].sum().reindex(weekday_order).reset_index()
            
            fig_weekday = px.bar(
                weekday_sales, x="Weekday", y="TotalPrice",
                title="Weekly Sales Pattern (Revenue by Weekday)",
                labels={"TotalPrice": "Revenue ($)", "Weekday": "Day of the Week"},
                color="TotalPrice", color_continuous_scale="Teal"
            )
            fig_weekday.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_weekday, use_container_width=True)
