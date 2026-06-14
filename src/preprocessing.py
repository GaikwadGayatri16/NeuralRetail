"""
Preprocessing module for NeuralRetail AI.
Cleans raw transaction data and saves it for downstream analytics.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Cleans and prepares raw Online Retail II Excel dataset.
    """
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
    def run(self) -> pd.DataFrame:
        """
        Executes the full preprocessing pipeline.
        """
        logger.info(f"Loading raw Excel file from {self.raw_data_path}")
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")
            
        # Load sheets
        logger.info("Reading sheet 'Year 2009-2010'...")
        df_2009 = pd.read_excel(self.raw_data_path, sheet_name="Year 2009-2010")
        logger.info(f"Loaded Year 2009-2010: {df_2009.shape} rows.")
        
        logger.info("Reading sheet 'Year 2010-2011'...")
        df_2010 = pd.read_excel(self.raw_data_path, sheet_name="Year 2010-2011")
        logger.info(f"Loaded Year 2010-2011: {df_2010.shape} rows.")
        
        # Concatenate
        df = pd.concat([df_2009, df_2010], ignore_index=True)
        logger.info(f"Combined dataset shape: {df.shape}")
        
        # Data Cleaning
        logger.info("Cleaning data...")
        
        # 1. Clean customer ID and remove missing ones
        initial_rows = len(df)
        df = df.dropna(subset=["Customer ID"])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing Customer ID.")
        
        # Ensure Customer ID is represented cleanly (int or string)
        df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
        
        # 2. Remove duplicates
        prev_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {prev_rows - len(df)} duplicate rows.")
        
        # 3. Remove cancelled invoices (starts with C)
        prev_rows = len(df)
        df = df[~df["Invoice"].astype(str).str.startswith("C", na=True)]
        logger.info(f"Removed {prev_rows - len(df)} cancelled transactions.")
        
        # 4. Remove negative or zero quantity
        prev_rows = len(df)
        df = df[df["Quantity"] > 0]
        logger.info(f"Removed {prev_rows - len(df)} rows with non-positive quantities.")
        
        # 5. Remove negative or zero price
        prev_rows = len(df)
        df = df[df["Price"] > 0]
        logger.info(f"Removed {prev_rows - len(df)} rows with non-positive prices.")
        
        # 6. Convert InvoiceDate to datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        
        # 7. Create TotalPrice column
        df["TotalPrice"] = df["Quantity"] * df["Price"]
        
        # Save output
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_data_path, index=False)
        logger.info(f"Preprocessed dataset saved to {self.processed_data_path}. Final shape: {df.shape}")
        
        return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    raw_path = BASE_DIR / "data" / "raw" / "online_retail_II.xlsx"
    processed_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
    
    preprocessor = DataPreprocessor(str(raw_path), str(processed_path))
    try:
        preprocessor.run()
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
