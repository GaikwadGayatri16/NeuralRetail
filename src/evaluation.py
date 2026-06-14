"""
Model Evaluation and Comparison module for NeuralRetail AI.
Consolidates performance metrics across customer segmentation, churn prediction,
and demand forecasting models, and saves the comparison results.
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Coordinates metrics extraction from all ML pipelines and compiles them.
    """
    def __init__(self, cleaned_path: str, feature_path: str, output_json_path: str):
        self.cleaned_path = Path(cleaned_path)
        self.feature_path = Path(feature_path)
        self.output_json_path = Path(output_json_path)
        
    def run(self):
        """
        Runs/loads all pipeline evaluations and compiles the metrics report.
        """
        logger.info("Initializing pipelines for evaluation...")
        
        # 1. Customer Segmentation Metrics
        # Import dynamically to avoid issues
        from src.segmentation import CustomerSegmenter
        model_dir = self.cleaned_path.parents[2] / "models"
        output_dir = self.cleaned_path.parents[2] / "outputs"
        
        seg_model_path = model_dir / "best_segmentation_model.pkl"
        seg_output_path = output_dir / "customer_segments.csv"
        
        segmenter = CustomerSegmenter(str(self.feature_path), str(seg_model_path), str(seg_output_path))
        logger.info("Running Customer Segmentation pipeline...")
        segmenter.run()
        seg_metrics = segmenter.metrics
        best_seg = segmenter.best_model_name
        
        # 2. Churn Prediction Metrics
        from src.churn import ChurnPredictor
        churn_model_path = model_dir / "best_churn_model.pkl"
        churn_output_path = output_dir / "churn_predictions.csv"
        
        predictor = ChurnPredictor(str(self.cleaned_path), str(self.feature_path), str(churn_model_path), str(churn_output_path))
        logger.info("Running Churn Prediction pipeline...")
        predictor.run()
        churn_metrics = predictor.metrics
        best_churn = predictor.best_model_name
        
        # 3. Demand Forecasting Metrics
        from src.forecasting import DemandForecaster
        forecast_model_path = model_dir / "best_forecast_model.pkl"
        forecast_output_path = output_dir / "sales_forecast.csv"
        
        forecaster = DemandForecaster(str(self.cleaned_path), str(forecast_model_path), str(forecast_output_path))
        logger.info("Running Demand Forecasting pipeline...")
        forecaster.run()
        forecast_metrics = forecaster.metrics
        best_forecast = forecaster.best_model_name
        
        # 4. Inventory Optimization (Run it to ensure outputs are fresh)
        from src.inventory import InventoryOptimizer
        abc_path = output_dir / "abc_analysis.csv"
        reorder_path = output_dir / "reorder_recommendations.csv"
        
        optimizer = InventoryOptimizer(str(self.cleaned_path), str(abc_path), str(reorder_path))
        logger.info("Running Inventory Optimization calculations...")
        optimizer.run()
        
        # Consolidate comparison report
        report = {
            "segmentation": {
                "metrics": seg_metrics,
                "best_model": best_seg
            },
            "churn": {
                "metrics": churn_metrics,
                "best_model": best_churn
            },
            "forecasting": {
                "metrics": forecast_metrics,
                "best_model": best_forecast
            }
        }
        
        # Save JSON
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Consolidated model comparison report saved to {self.output_json_path}")
        return report

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    cleaned_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
    feature_path = BASE_DIR / "data" / "processed" / "feature_store.csv"
    comparison_path = BASE_DIR / "outputs" / "model_comparison.json"
    
    evaluator = ModelEvaluator(str(cleaned_path), str(feature_path), str(comparison_path))
    try:
        evaluator.run()
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
