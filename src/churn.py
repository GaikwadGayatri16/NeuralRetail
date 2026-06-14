"""
Churn Prediction module for NeuralRetail AI.
Defines churn, builds feature-label dataset using temporal split to prevent leakage,
trains/compares classifiers, and saves predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """
    Fits and compares classification models for customer churn prediction.
    """
    def __init__(self, cleaned_data_path: str, feature_store_path: str, model_path: str, output_path: str):
        self.cleaned_data_path = Path(cleaned_data_path)
        self.feature_store_path = Path(feature_store_path)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    def prepare_leakage_free_data(self, df: pd.DataFrame) -> tuple:
        """
        Creates a training dataset by splitting the data temporally.
        Observation window: transactions before (T_max - 90 days).
        Label window: last 90 days of transactions.
        """
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        t_max = df["InvoiceDate"].max()
        cutoff_date = t_max - timedelta(days=90)
        logger.info(f"Temporal Split: Max Date={t_max.date()}, Cutoff Date={cutoff_date.date()}")
        
        # 1. Transactions in observation and label windows
        obs_df = df[df["InvoiceDate"] < cutoff_date].copy()
        label_df = df[df["InvoiceDate"] >= cutoff_date].copy()
        
        # If observation data is empty, we cannot train
        if obs_df.empty or label_df.empty:
            raise ValueError("Insufficient data to perform a 90-day temporal split.")
            
        # 2. Get active customers in the observation window
        obs_customers = obs_df["Customer ID"].unique()
        logger.info(f"Active customers in observation window: {len(obs_customers)}")
        
        # 3. Label: Churn = 1 if customer did NOT purchase in the label window, else 0
        active_in_label = set(label_df["Customer ID"].unique())
        labels = {cust: (0 if cust in active_in_label else 1) for cust in obs_customers}
        
        # 4. Features for the observation window
        snapshot_date = cutoff_date
        features = obs_df.groupby("Customer ID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
            TotalOrders=("Invoice", "nunique"),
            TotalProducts=("Quantity", "sum"),
            AverageOrderValue=("TotalPrice", "mean"),
            LifetimeRevenue=("TotalPrice", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max")
        ).reset_index()
        
        features["CustomerAgeDays"] = (features["LastPurchase"] - features["FirstPurchase"]).dt.days
        features["DaysSinceLastPurchase"] = (snapshot_date - features["LastPurchase"]).dt.days
        features = features.drop(columns=["FirstPurchase", "LastPurchase"])
        
        # Map labels
        features["Churn"] = features["Customer ID"].map(labels)
        
        return features

    def run(self):
        """
        Trains classifiers, evaluates, selects the best model, and saves predictions.
        """
        logger.info(f"Loading cleaned transactions from {self.cleaned_data_path}")
        if not self.cleaned_data_path.exists():
            raise FileNotFoundError(f"Cleaned transactions not found at {self.cleaned_data_path}")
            
        df_clean = pd.read_csv(self.cleaned_data_path)
        
        # Prepare leakage-free training data
        train_data = self.prepare_leakage_free_data(df_clean)
        
        # Features and target
        feature_cols = [
            "Recency", "Frequency", "Monetary", "TotalOrders", "TotalProducts",
            "AverageOrderValue", "LifetimeRevenue", "CustomerAgeDays", "DaysSinceLastPurchase"
        ]
        
        X = train_data[feature_cols]
        y = train_data["Churn"]
        
        logger.info(f"Training dataset shape: {X.shape}, Class distribution: {y.value_counts(normalize=True).to_dict()}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            "XGBoost": XGBClassifier(random_state=42, n_estimators=100, max_depth=5, eval_metric="logloss")
        }
        
        model_metrics = {}
        fitted_models = {}
        
        # Train and evaluate
        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_prob)
                
                model_metrics[name] = {
                    "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "ROC AUC": roc_auc
                }
                fitted_models[name] = model
                logger.info(f"{name} Results: F1={f1:.4f}, ROC AUC={roc_auc:.4f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                
        # Select best model based on F1 Score
        self.best_model_name = max(model_metrics, key=lambda k: model_metrics[k]["F1 Score"])
        self.best_model = fitted_models[self.best_model_name]
        self.metrics = model_metrics
        
        logger.info(f"Automatically selected best churn model: {self.best_model_name}")
        
        # Final train of the best model on all observation data to refine weights
        X_full_scaled = scaler.fit_transform(X)
        self.best_model.fit(X_full_scaled, y)
        
        # Save model and scaler
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.best_model, "scaler": scaler, "model_name": self.best_model_name, "features": feature_cols}, self.model_path)
        logger.info(f"Best churn model and scaler saved to {self.model_path}")
        
        # --------------------------
        # GENERATE CURRENT PREDICTIONS
        # --------------------------
        # For predicting actual churn risk today, we apply the model on the full current features
        logger.info(f"Loading current features from {self.feature_store_path}")
        current_features_df = pd.read_csv(self.feature_store_path)
        
        X_current = current_features_df[feature_cols]
        X_current_scaled = scaler.transform(X_current)
        
        current_features_df["ChurnProbability"] = self.best_model.predict_proba(X_current_scaled)[:, 1]
        # Label churn if DaysSinceLastPurchase > 90 (business definition) or if predicted prob > 0.5
        # The prompt says: "Define churn: Customer inactive for more than 90 days."
        # So we can use the business definition for ground truth today, and let the model predict the probability.
        current_features_df["IsChurned_GroundTruth"] = (current_features_df["DaysSinceLastPurchase"] > 90).astype(int)
        current_features_df["PredictedChurn"] = (current_features_df["ChurnProbability"] > 0.5).astype(int)
        
        # Save output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        current_features_df.to_csv(self.output_path, index=False)
        logger.info(f"Churn predictions saved to {self.output_path}")
        
        # Print summary stats
        logger.info(f"Ground Truth Churn rate: {current_features_df['IsChurned_GroundTruth'].mean():.2%}")
        logger.info(f"Predicted Churn rate: {current_features_df['PredictedChurn'].mean():.2%}")
        
        return current_features_df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    cleaned_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
    feature_path = BASE_DIR / "data" / "processed" / "feature_store.csv"
    model_path = BASE_DIR / "models" / "best_churn_model.pkl"
    output_path = BASE_DIR / "outputs" / "churn_predictions.csv"
    
    predictor = ChurnPredictor(str(cleaned_path), str(feature_path), str(model_path), str(output_path))
    try:
        predictor.run()
    except Exception as e:
        logger.error(f"Churn script failed: {e}", exc_info=True)
