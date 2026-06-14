"""
Demand Forecasting module for NeuralRetail AI.
Aggregates sales by day, builds time-series lag/rolling features,
compares Prophet and XGBoost, and generates a 30-day forecast.
Includes a fallback to Holt-Winters Exponential Smoothing if Prophet is unavailable.
"""

import os
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try importing prophet, else define fallback flag
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. A Statsmodels Holt-Winters fallback model will be used in place of Prophet.")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

class DemandForecaster:
    """
    Trains and compares time-series forecasting models.
    """
    def __init__(self, cleaned_data_path: str, model_path: str, output_path: str):
        self.cleaned_data_path = Path(cleaned_data_path)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.best_model_name = None
        self.best_model = None
        self.metrics = {}

    def prepare_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates sales by date and handles missing dates.
        """
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Date"] = df["InvoiceDate"].dt.date
        
        # Aggregate daily sales revenue
        daily_sales = df.groupby("Date")["TotalPrice"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]
        daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])
        
        # Sort and fill missing dates with 0 sales
        daily_sales = daily_sales.sort_values(by="ds").set_index("ds")
        all_dates = pd.date_range(start=daily_sales.index.min(), end=daily_sales.index.max(), freq="D")
        daily_sales = daily_sales.reindex(all_dates, fill_value=0.0).reset_index()
        daily_sales.columns = ["ds", "y"]
        
        return daily_sales

    def add_xgboost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag, rolling mean, and calendar features for XGBoost.
        """
        df_feat = df.copy()
        
        # Lags
        df_feat["Lag_1"] = df_feat["y"].shift(1)
        df_feat["Lag_7"] = df_feat["y"].shift(7)
        df_feat["Lag_30"] = df_feat["y"].shift(30)
        
        # Rolling Means (shift by 1 to prevent leakage)
        df_feat["Rolling_Mean_7"] = df_feat["y"].shift(1).rolling(window=7).mean()
        df_feat["Rolling_Mean_30"] = df_feat["y"].shift(1).rolling(window=30).mean()
        
        # Calendar Features
        df_feat["Month"] = df_feat["ds"].dt.month
        df_feat["Weekday"] = df_feat["ds"].dt.weekday
        df_feat["Quarter"] = df_feat["ds"].dt.quarter
        
        return df_feat

    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Mean Absolute Percentage Error (MAPE).
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero by setting a small epsilon or filtering
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def fit_fallback_prophet(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        Fits Holt-Winters Exponential Smoothing as a fallback for Prophet.
        """
        logger.info("Fitting Holt-Winters Exponential Smoothing model...")
        train_series = train_df.set_index("ds")["y"]
        # Fit model
        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=7
        ).fit()
        
        # Predict on test set
        preds = model.forecast(len(test_df))
        return model, preds.values

    def run(self):
        """
        Runs the forecasting pipeline: preparation, feature engineering, split, training, selection, and forecasting.
        """
        logger.info(f"Loading cleaned retail data from {self.cleaned_data_path}")
        if not self.cleaned_data_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {self.cleaned_data_path}")
            
        df = pd.read_csv(self.cleaned_data_path)
        daily_sales = self.prepare_time_series(df)
        
        if len(daily_sales) < 60:
            raise ValueError("Insufficient daily data points to perform training and forecasting.")
            
        # Split: last 30 days as test
        train_df = daily_sales.iloc[:-30].copy()
        test_df = daily_sales.iloc[-30:].copy()
        
        # --------------------------
        # 1. TRAIN PROPHET / FALLBACK
        # --------------------------
        prophet_preds = None
        prophet_model = None
        
        if PROPHET_AVAILABLE:
            logger.info("Training Prophet model...")
            try:
                # Fit Prophet
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                m.fit(train_df)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                prophet_preds = forecast.iloc[-30:]["yhat"].values
                prophet_model = m
            except Exception as e:
                logger.error(f"Prophet training failed: {e}. Falling back to Holt-Winters.")
                prophet_model, prophet_preds = self.fit_fallback_prophet(train_df, test_df)
        else:
            prophet_model, prophet_preds = self.fit_fallback_prophet(train_df, test_df)
            
        # Evaluate Prophet/Fallback
        p_rmse = np.sqrt(mean_squared_error(test_df["y"], prophet_preds))
        p_mae = mean_absolute_error(test_df["y"], prophet_preds)
        p_mape = self.calculate_mape(test_df["y"], prophet_preds)
        
        p_name = "Prophet" if PROPHET_AVAILABLE else "Holt-Winters"
        self.metrics[p_name] = {"RMSE": p_rmse, "MAE": p_mae, "MAPE": p_mape}
        logger.info(f"{p_name} Test Metrics: RMSE={p_rmse:.2f}, MAE={p_mae:.2f}, MAPE={p_mape:.2f}%")
        
        # --------------------------
        # 2. TRAIN XGBOOST
        # --------------------------
        logger.info("Training XGBoost Time Series model...")
        df_xgb_feat = self.add_xgboost_features(daily_sales)
        
        # Drop NaN rows due to lags/rolling averages (first 30 rows)
        df_xgb_clean = df_xgb_feat.dropna().copy()
        
        # Split temporally based on dates
        test_dates = test_df["ds"]
        train_xgb = df_xgb_clean[~df_xgb_clean["ds"].isin(test_dates)]
        test_xgb = df_xgb_clean[df_xgb_clean["ds"].isin(test_dates)]
        
        feature_cols = [
            "Lag_1", "Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30",
            "Month", "Weekday", "Quarter"
        ]
        
        from xgboost import XGBRegressor
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        
        xgb_preds = None
        if not train_xgb.empty and not test_xgb.empty:
            xgb_model.fit(train_xgb[feature_cols], train_xgb["y"])
            xgb_preds = xgb_model.predict(test_xgb[feature_cols])
            
            # Evaluate XGBoost
            x_rmse = np.sqrt(mean_squared_error(test_xgb["y"], xgb_preds))
            x_mae = mean_absolute_error(test_xgb["y"], xgb_preds)
            x_mape = self.calculate_mape(test_xgb["y"], xgb_preds)
            self.metrics["XGBoost"] = {"RMSE": x_rmse, "MAE": x_mae, "MAPE": x_mape}
            logger.info(f"XGBoost Test Metrics: RMSE={x_rmse:.2f}, MAE={x_mae:.2f}, MAPE={x_mape:.2f}%")
        else:
            logger.warning("XGBoost training data empty. Scoring set to infinity.")
            self.metrics["XGBoost"] = {"RMSE": 999999.0, "MAE": 999999.0, "MAPE": 100.0}
            
        # --------------------------
        # 3. SELECT BEST MODEL
        # --------------------------
        self.best_model_name = min(self.metrics, key=lambda k: self.metrics[k]["RMSE"])
        logger.info(f"Automatically selected best forecast model: {self.best_model_name}")
        
        # Fit best model on full dataset
        if self.best_model_name in ["Prophet", "Holt-Winters"]:
            if self.best_model_name == "Prophet":
                self.best_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                self.best_model.fit(daily_sales)
                future = self.best_model.make_future_dataframe(periods=30)
                forecast = self.best_model.predict(future)
                
                # Align actuals and forecast
                historical_forecast = forecast.iloc[:-30][["ds", "yhat"]].rename(columns={"yhat": "Predicted"})
                future_forecast = forecast.iloc[-30:][["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "Predicted"})
            else:  # Holt-Winters
                series = daily_sales.set_index("ds")["y"]
                self.best_model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7).fit()
                future_ds = pd.date_range(start=daily_sales["ds"].max() + pd.Timedelta(days=1), periods=30, freq="D")
                preds_future = self.best_model.forecast(30)
                
                historical_forecast = pd.DataFrame({
                    "ds": daily_sales["ds"],
                    "Predicted": self.best_model.fittedvalues.values
                })
                future_forecast = pd.DataFrame({
                    "ds": future_ds,
                    "Predicted": preds_future.values,
                    "yhat_lower": preds_future.values * 0.85, # Simulated confidence intervals for Holt-Winters
                    "yhat_upper": preds_future.values * 1.15
                })
        else:  # XGBoost
            # Train on full data
            xgb_model.fit(df_xgb_clean[feature_cols], df_xgb_clean["y"])
            self.best_model = xgb_model
            
            # Predict historical (retaining NaN rows as NaN)
            hist_preds = np.nan * np.ones(len(daily_sales))
            hist_preds[30:] = xgb_model.predict(df_xgb_clean[feature_cols])
            
            historical_forecast = pd.DataFrame({
                "ds": daily_sales["ds"],
                "Predicted": hist_preds
            })
            
            # Recursive future forecasting for 30 days
            future_preds = []
            future_dates = pd.date_range(start=daily_sales["ds"].max() + pd.Timedelta(days=1), periods=30, freq="D")
            
            temp_df = daily_sales.copy()
            for i in range(30):
                new_row = pd.DataFrame({"ds": [future_dates[i]], "y": [0.0]})
                temp_df = pd.concat([temp_df, new_row], ignore_index=True)
                # Recalculate features for the last row
                feat_df = self.add_xgboost_features(temp_df)
                pred_val = xgb_model.predict(feat_df[feature_cols].iloc[[-1]])[0]
                pred_val = max(0.0, pred_val) # Prevent negative sales
                temp_df.loc[temp_df.index[-1], "y"] = pred_val
                future_preds.append(pred_val)
                
            future_forecast = pd.DataFrame({
                "ds": future_dates,
                "Predicted": future_preds,
                "yhat_lower": np.array(future_preds) * 0.85,
                "yhat_upper": np.array(future_preds) * 1.15
            })
            
        # Combine into a single export dataset
        # Historical actuals + predicted
        historical_export = pd.merge(daily_sales, historical_forecast, on="ds", how="left")
        historical_export["Type"] = "Historical"
        historical_export["yhat_lower"] = historical_export["y"]
        historical_export["yhat_upper"] = historical_export["y"]
        
        # Future forecast
        future_export = future_forecast.copy()
        future_export["y"] = np.nan
        future_export["Type"] = "Forecast"
        
        forecast_df = pd.concat([historical_export, future_export], ignore_index=True)
        
        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.best_model,
            "model_name": self.best_model_name,
            "metrics": self.metrics,
            "prophet_available": PROPHET_AVAILABLE
        }, self.model_path)
        logger.info(f"Best forecast model saved to {self.model_path}")
        
        # Save output CSV
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(self.output_path, index=False)
        logger.info(f"Demand forecast output saved to {self.output_path}. Shape: {forecast_df.shape}")
        
        return forecast_df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    cleaned_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
    model_path = BASE_DIR / "models" / "best_forecast_model.pkl"
    output_path = BASE_DIR / "outputs" / "sales_forecast.csv"
    
    forecaster = DemandForecaster(str(cleaned_path), str(model_path), str(output_path))
    try:
        forecaster.run()
    except Exception as e:
        logger.error(f"Forecasting script failed: {e}", exc_info=True)
