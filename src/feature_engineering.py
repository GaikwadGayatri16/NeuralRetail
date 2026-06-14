"""
Feature engineering module for NeuralRetail AI.
Calculates customer-level RFM and behavioral metrics.
"""

import pandas as pd
from datetime import timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Computes RFM and customer behavioral features.
    """
    def __init__(self, cleaned_data_path: str, feature_store_path: str):
        self.cleaned_data_path = Path(cleaned_data_path)
        self.feature_store_path = Path(feature_store_path)
        
    def run(self) -> pd.DataFrame:
        """
        Executes the feature engineering pipeline.
        """
        logger.info(f"Loading cleaned dataset from {self.cleaned_data_path}")
        if not self.cleaned_data_path.exists():
            raise FileNotFoundError(f"Cleaned data file not found at {self.cleaned_data_path}")
            
        df = pd.read_csv(self.cleaned_data_path)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        
        # 1. Establish reference date (max date + 1 day)
        snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)
        logger.info(f"Using snapshot date: {snapshot_date}")
        
        # 2. RFM Aggregation
        rfm = df.groupby("Customer ID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum")
        ).reset_index()
        
        # 3. Behavioral Features Aggregation
        behavioral = df.groupby("Customer ID").agg(
            TotalOrders=("Invoice", "nunique"),
            TotalProducts=("Quantity", "sum"),
            AverageOrderValue=("TotalPrice", "mean"),
            LifetimeRevenue=("TotalPrice", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max")
        ).reset_index()
        
        # 4. Age and time since purchase features
        behavioral["CustomerAgeDays"] = (behavioral["LastPurchase"] - behavioral["FirstPurchase"]).dt.days
        behavioral["DaysSinceLastPurchase"] = (snapshot_date - behavioral["LastPurchase"]).dt.days
        
        # Drop temporary timestamps
        behavioral = behavioral.drop(columns=["FirstPurchase", "LastPurchase"])
        
        # 5. Merge datasets
        feature_store = pd.merge(rfm, behavioral, on="Customer ID")
        
        # Save output
        self.feature_store_path.parent.mkdir(parents=True, exist_ok=True)
        feature_store.to_csv(self.feature_store_path, index=False)
        logger.info(f"Feature store successfully saved to {self.feature_store_path}. Shape: {feature_store.shape}")
        
        return feature_store

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    cleaned_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
    feature_path = BASE_DIR / "data" / "processed" / "feature_store.csv"
    
    engineer = FeatureEngineer(str(cleaned_path), str(feature_path))
    try:
        engineer.run()
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
