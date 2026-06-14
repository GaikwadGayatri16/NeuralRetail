"""
Inventory Optimization module for NeuralRetail AI.
Performs ABC analysis, safety stock calculations, reorder point estimations,
EOQ recommendations, and moving velocity categorization.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class InventoryOptimizer:
    """
    Computes retail inventory metrics and reorder recommendations.
    """
    def __init__(self, cleaned_data_path: str, abc_path: str, reorder_path: str):
        self.cleaned_data_path = Path(cleaned_data_path)
        self.abc_path = Path(abc_path)
        self.reorder_path = Path(reorder_path)
        
    def run(self):
        """
        Runs the inventory optimization pipeline.
        """
        logger.info(f"Loading cleaned retail data from {self.cleaned_data_path}")
        if not self.cleaned_data_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {self.cleaned_data_path}")
            
        df = pd.read_csv(self.cleaned_data_path)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Date"] = df["InvoiceDate"].dt.date
        
        # 1. Product Sales Aggregations
        logger.info("Aggregating sales by product...")
        product_sales = df.groupby(["StockCode", "Description"]).agg(
            TotalRevenue=("TotalPrice", "sum"),
            TotalQuantity=("Quantity", "sum"),
            TransactionCount=("Invoice", "nunique")
        ).reset_index()
        
        # 2. ABC Analysis
        logger.info("Performing ABC Analysis...")
        product_sales = product_sales.sort_values(by="TotalRevenue", ascending=False).reset_index(drop=True)
        total_revenue = product_sales["TotalRevenue"].sum()
        product_sales["RevenueContribution"] = product_sales["TotalRevenue"] / total_revenue
        product_sales["CumulativeRevenueShare"] = product_sales["RevenueContribution"].cumsum()
        
        def classify_abc(cum_share):
            if cum_share <= 0.80:
                return "A"
            elif cum_share <= 0.95:
                return "B"
            else:
                return "C"
                
        product_sales["ABC_Class"] = product_sales["CumulativeRevenueShare"].apply(classify_abc)
        
        # Save ABC Analysis
        self.abc_path.parent.mkdir(parents=True, exist_ok=True)
        product_sales.to_csv(self.abc_path, index=False)
        logger.info(f"ABC Analysis saved to {self.abc_path}. Shape: {product_sales.shape}")
        
        # 3. Daily quantity metrics per product for safety stock & ROP
        logger.info("Calculating daily demand distribution per product...")
        # Get daily quantity sold for each product
        daily_product_qty = df.groupby(["StockCode", "Description", "Date"])["Quantity"].sum().reset_index()
        
        # Group by product to get mean and max daily sales
        daily_stats = daily_product_qty.groupby(["StockCode", "Description"]).agg(
            AvgDailyQty=("Quantity", "mean"),
            MaxDailyQty=("Quantity", "max"),
            StdDailyQty=("Quantity", "std")
        ).reset_index()
        
        daily_stats["StdDailyQty"] = daily_stats["StdDailyQty"].fillna(0.0)
        
        # Merge stats with product sales
        inventory_df = pd.merge(product_sales, daily_stats, on=["StockCode", "Description"])
        
        # 4. Inventory Parameters (Safety Stock, ROP, EOQ)
        logger.info("Computing Safety Stock, Reorder Points, and Economic Order Quantities...")
        # Parameters: Lead time (days)
        avg_lead_time = 7
        max_lead_time = 14
        
        # Standard safety stock formula: (Max Daily Qty * Max Lead Time) - (Avg Daily Qty * Avg Lead Time)
        inventory_df["SafetyStock"] = (
            (inventory_df["MaxDailyQty"] * max_lead_time) - 
            (inventory_df["AvgDailyQty"] * avg_lead_time)
        )
        # Ensure Safety Stock is non-negative
        inventory_df["SafetyStock"] = inventory_df["SafetyStock"].apply(lambda x: max(0.0, x)).round(1)
        
        # Reorder Point (ROP) = (Avg Daily Qty * Avg Lead Time) + Safety Stock
        inventory_df["ReorderPoint"] = (
            (inventory_df["AvgDailyQty"] * avg_lead_time) + 
            inventory_df["SafetyStock"]
        ).round(1)
        
        # Simulating current stock level for demo (deterministic based on description length to make it interesting)
        # We simulate current stock to be between 0.3x and 1.8x of the ROP
        np.random.seed(42)
        random_factors = 0.3 + 1.5 * (inventory_df["Description"].str.len() % 10) / 10.0
        inventory_df["CurrentStock"] = (inventory_df["ReorderPoint"] * random_factors).round(0)
        
        # Suggest Economic Order Quantity (EOQ)
        # EOQ = sqrt((2 * AnnualDemand * OrderingCost) / HoldingCost)
        # Ordering Cost = $15, Holding Cost = $1.5 per unit/year
        ordering_cost = 15.0
        holding_cost = 1.5
        inventory_df["AnnualDemand"] = inventory_df["AvgDailyQty"] * 365
        inventory_df["EOQ"] = np.sqrt(
            (2 * inventory_df["AnnualDemand"] * ordering_cost) / holding_cost
        ).round(0)
        
        # 5. Reorder Recommendations
        inventory_df["ReorderRecommended"] = (inventory_df["CurrentStock"] <= inventory_df["ReorderPoint"]).astype(int)
        
        # Sort by total revenue or quantity sold
        inventory_df = inventory_df.sort_values(by="TotalRevenue", ascending=False).reset_index(drop=True)
        
        # 6. Velocity Categorization (Fast / Slow Moving)
        # Fast Moving: top 10% of products by daily velocity
        # Slow Moving: bottom 10% of products by daily velocity
        q_high = inventory_df["AvgDailyQty"].quantile(0.90)
        q_low = inventory_df["AvgDailyQty"].quantile(0.10)
        
        def classify_velocity(avg_qty):
            if avg_qty >= q_high:
                return "Fast Moving"
            elif avg_qty <= q_low:
                return "Slow Moving"
            else:
                return "Normal"
                
        inventory_df["Velocity"] = inventory_df["AvgDailyQty"].apply(classify_velocity)
        
        # Select key columns for reorder recommendations
        reorder_cols = [
            "StockCode", "Description", "ABC_Class", "Velocity", "AvgDailyQty",
            "SafetyStock", "ReorderPoint", "CurrentStock", "EOQ", "ReorderRecommended"
        ]
        reorder_df = inventory_df[reorder_cols].copy()
        
        # Save Reorder Recommendations
        self.reorder_path.parent.mkdir(parents=True, exist_ok=True)
        reorder_df.to_csv(self.reorder_path, index=False)
        logger.info(f"Reorder recommendations saved to {self.reorder_path}. Shape: {reorder_df.shape}")
        
        # Print summary stats
        logger.info(f"ABC Class Breakdown:\n{reorder_df['ABC_Class'].value_counts().to_string()}")
        logger.info(f"Reorders Recommended: {reorder_df['ReorderRecommended'].sum()} products")
        
        return reorder_df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    cleaned_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
    abc_path = BASE_DIR / "outputs" / "abc_analysis.csv"
    reorder_path = BASE_DIR / "outputs" / "reorder_recommendations.csv"
    
    optimizer = InventoryOptimizer(str(cleaned_path), str(abc_path), str(reorder_path))
    try:
        optimizer.run()
    except Exception as e:
        logger.error(f"Inventory script failed: {e}", exc_info=True)
