"""
Customer Segmentation module for NeuralRetail AI.
Trains and compares KMeans, GMM, and DBSCAN on RFM features,
selects the best model, maps clusters to business segments, and saves outputs.
"""

import os
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CustomerSegmenter:
    """
    Fits and compares clustering models for RFM segmentation.
    """
    def __init__(self, feature_store_path: str, model_path: str, output_path: str):
        self.feature_store_path = Path(feature_store_path)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.best_model_name = None
        self.best_model = None
        self.metrics = {}

    def prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applies log-transformation and StandardScaler to RFM features.
        """
        # Select RFM features
        rfm_features = df[["Recency", "Frequency", "Monetary"]].copy()
        
        # Apply log transformation to handle skewness
        rfm_log = np.log1p(rfm_features)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(rfm_log)
        return X_scaled

    def run(self):
        """
        Runs the segmentation training, evaluation, selection, and mapping.
        """
        logger.info(f"Loading feature store from {self.feature_store_path}")
        if not self.feature_store_path.exists():
            raise FileNotFoundError(f"Feature store not found at {self.feature_store_path}")
            
        df = pd.read_csv(self.feature_store_path)
        
        # We need at least some rows to cluster
        if len(df) < 5:
            raise ValueError("Insufficient data in feature store to perform clustering.")
            
        X_scaled = self.prepare_data(df)
        
        # Initialize models
        models = {
            "KMeans": KMeans(n_clusters=4, random_state=42, n_init=10),
            "GMM": GaussianMixture(n_components=4, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }
        
        model_scores = {}
        fitted_labels = {}
        
        # Train and evaluate models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                if name == "GMM":
                    labels = model.fit_predict(X_scaled)
                elif name == "DBSCAN":
                    labels = model.fit_predict(X_scaled)
                else:  # KMeans
                    labels = model.fit_predict(X_scaled)
                    
                # Calculate scores (ignore noise in DBSCAN for silhouette score calculation if it exists)
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                
                if n_clusters < 2:
                    logger.warning(f"{name} produced less than 2 clusters ({n_clusters}). Setting score to -1.")
                    sil, db = -1.0, 999.0
                else:
                    # Filter noise for silhouette score
                    mask = labels != -1
                    if mask.sum() > 10:
                        sil = silhouette_score(X_scaled[mask], labels[mask])
                        db = davies_bouldin_score(X_scaled[mask], labels[mask])
                    else:
                        sil, db = -1.0, 999.0
                        
                model_scores[name] = {"Silhouette": sil, "Davies-Bouldin": db}
                fitted_labels[name] = labels
                logger.info(f"{name} Results: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                model_scores[name] = {"Silhouette": -1.0, "Davies-Bouldin": 999.0}
                fitted_labels[name] = np.zeros(len(df))
                
        # Select best model based on Silhouette Score
        self.best_model_name = max(model_scores, key=lambda k: model_scores[k]["Silhouette"])
        self.best_model = models[self.best_model_name]
        self.metrics = model_scores
        
        logger.info(f"Automatically selected best model: {self.best_model_name}")
        
        # Save best model and standard scaler together
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.best_model, "scaler": self.scaler, "model_name": self.best_model_name}, self.model_path)
        logger.info(f"Best segmentation model and scaler saved to {self.model_path}")
        
        # Add labels from the best model
        df["Cluster"] = fitted_labels[self.best_model_name]
        
        # Map clusters to segments
        df = self.map_clusters_to_segments(df)
        
        # Save customer segments CSV
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Segmented customer profiles saved to {self.output_path}")
        
        # Print summary stats
        logger.info("\nSegment Distribution:")
        logger.info(df["Segment"].value_counts().to_string())
        
        return df

    def map_clusters_to_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps cluster IDs to customer segment labels (VIP, Loyal, Regular, At Risk)
        using a custom behavioral score derived from cluster centroids.
        """
        # If the best model is DBSCAN, we may have noise or arbitrary cluster counts.
        # To ensure we have exactly 4 segments, we fall back to a rule-based ranking
        # or map whatever clusters we have to the 4 categories based on monetary value.
        
        unique_clusters = df["Cluster"].unique()
        
        # Filter out noise label (-1) for centroid calculation
        valid_clusters = [c for c in unique_clusters if c != -1]
        
        # If DBSCAN is chosen and finds 1 cluster or only noise, fallback to standard rule-based
        if len(valid_clusters) == 0:
            df["Segment"] = "At Risk Customers"
            return df
            
        # Compute cluster centroids (mean of Recency, Frequency, Monetary)
        centroids = df[df["Cluster"] != -1].groupby("Cluster").agg(
            R_mean=("Recency", "mean"),
            F_mean=("Frequency", "mean"),
            M_mean=("Monetary", "mean")
        ).reset_index()
        
        # Scale centroid means to 0-1 to compute behavioral score
        r_min, r_max = centroids["R_mean"].min(), centroids["R_mean"].max()
        f_min, f_max = centroids["F_mean"].min(), centroids["F_mean"].max()
        m_min, m_max = centroids["M_mean"].min(), centroids["M_mean"].max()
        
        # Small epsilon to avoid divide by zero
        eps = 1e-6
        centroids["R_norm"] = (centroids["R_mean"] - r_min) / (r_max - r_min + eps)
        centroids["F_norm"] = (centroids["F_mean"] - f_min) / (f_max - f_min + eps)
        centroids["M_norm"] = (centroids["M_mean"] - m_min) / (m_max - m_min + eps)
        
        # Behavioral score: high frequency, high monetary, low recency
        centroids["Score"] = -centroids["R_norm"] + centroids["F_norm"] + centroids["M_norm"]
        
        # Sort clusters by Score
        centroids_sorted = centroids.sort_values(by="Score").reset_index(drop=True)
        
        # Assign segments based on rank
        # We map ranks to: 0 -> At Risk, 1 -> Regular, 2 -> Loyal, 3 -> VIP
        num_clusters = len(centroids_sorted)
        cluster_to_segment = {}
        
        # Handle cases with different cluster numbers
        if num_clusters >= 4:
            for idx, row in centroids_sorted.iterrows():
                c_id = int(row["Cluster"])
                if idx == 0:
                    cluster_to_segment[c_id] = "At Risk Customers"
                elif idx == num_clusters - 1:
                    cluster_to_segment[c_id] = "VIP Customers"
                elif idx == num_clusters - 2:
                    cluster_to_segment[c_id] = "Loyal Customers"
                else:
                    cluster_to_segment[c_id] = "Regular Customers"
        else:
            # Fallback for fewer clusters
            segments_list = ["At Risk Customers", "Regular Customers", "Loyal Customers", "VIP Customers"]
            for idx, row in centroids_sorted.iterrows():
                c_id = int(row["Cluster"])
                # Distribute evenly
                segment_idx = int((idx / num_clusters) * len(segments_list))
                cluster_to_segment[c_id] = segments_list[min(segment_idx, len(segments_list) - 1)]
                
        # Handle DBSCAN noise if present
        cluster_to_segment[-1] = "At Risk Customers"
        
        # Map segments
        df["Segment"] = df["Cluster"].map(cluster_to_segment)
        return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    feature_path = BASE_DIR / "data" / "processed" / "feature_store.csv"
    model_path = BASE_DIR / "models" / "best_segmentation_model.pkl"
    output_path = BASE_DIR / "outputs" / "customer_segments.csv"
    
    segmenter = CustomerSegmenter(str(feature_path), str(model_path), str(output_path))
    try:
        segmenter.run()
    except Exception as e:
        logger.error(f"Segmentation script failed: {e}", exc_info=True)
