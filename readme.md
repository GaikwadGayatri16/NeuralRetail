# NeuralRetail AI – Intelligent Retail Analytics Dashboard

NeuralRetail AI is an end-to-end retail intelligence and machine learning analytics platform. It uses the UCI **Online Retail II** dataset to analyze transactional history, segment customers, predict shopper churn, forecast future sales demand, and optimize product inventory. 

This is a academic final-year project designed for a collaborative team of 6 students divided into 3 specialized focus groups.

---

## 👥 Team & Sub-Group Structure

### 👥 Group 1: Customer Intelligence Team
* **Responsibility**: Customer segmentation and profiling.
* **Core Operations**: RFM metrics generation, clustering models comparison (KMeans, DBSCAN, Gaussian Mixture Model), and marketing persona generation.

### 👥 Group 2: Churn Analytics Team
* **Responsibility**: Shopper churn forecasting.
* **Core Operations**: Leakage-free temporal training, churn prediction (Logistic Regression, Random Forest, XGBoost), feature importance extraction, and target marketing list.

### 👥 Group 3: Demand Forecasting & Inventory Team
* **Responsibility**: Sales planning and catalog stock levels.
* **Core Operations**: Daily sales forecasting (Prophet, XGBoost), ABC product classification, safety stock, and Reorder Point (ROP/EOQ) recommendations.

---

## 📂 Project Structure

```
NeuralRetail/
│
├── app.py                      # Main landing page for Streamlit
│
├── data/
│   ├── raw/                    # Contains online_retail_II.xlsx
│   └── processed/              # Contains cleaned_retail.csv, feature_store.csv
│
├── models/                     # Best serialized models (.pkl)
│
├── outputs/                    # Output CSVs and metric JSONs
│
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory Data Analysis & visual plots
│
├── src/
│   ├── preprocessing.py        # Raw Excel cleaning & formatting
│   ├── feature_engineering.py  # RFM and advanced behavioral extraction
│   ├── segmentation.py         # Customer clustering pipelines
│   ├── churn.py                # Classifier models for customer retention
│   ├── forecasting.py          # Daily sales time-series forecasting
│   ├── inventory.py            # ABC, safety stock, ROP, EOQ optimizer
│   └── evaluation.py           # Coordinator script compiles all model metrics
│
├── pages/                      # Multi-page dashboard pages
│   ├── 1_Executive_Dashboard.py
│   ├── 2_Sales_Analytics.py
│   ├── 3_Customer_Segmentation.py
│   ├── 4_Churn_Prediction.py
│   ├── 5_Demand_Forecasting.py
│   ├── 6_Inventory_Optimization.py
│   └── 7_Model_Comparison.py
│
├── requirements.txt            # Project python dependencies
├── README.md                   # Setup and system manual
└── .gitignore                  # Git exclude config
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.11** (or 3.9+) installed on your local computer.

### 2. Install Dependencies
Clone this repository or open the project folder, open a terminal, and install the required libraries:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Machine Learning Pipelines

To prepare the data, train the models, and generate the outputs, run the scripts in sequence:

### Step 1: Preprocessing & Cleaning
Loads raw Excel spreadsheets, merges sheets, handles invalid transactions, and exports clean CSV:
```bash
python src/preprocessing.py
```

### Step 2: Feature Engineering
Calculates customer-level RFM features and advanced purchase behaviors:
```bash
python src/feature_engineering.py
```

### Step 3: Run Model Pipelines & Evaluation
Run the central coordinator script to execute all models, run clustering/classification/forecasting, pick the best models, and export comparative reports:
```bash
python src/evaluation.py
```

*Note: You can also run individual model scripts if preferred:*
* `python src/segmentation.py`
* `python src/churn.py`
* `python src/forecasting.py`
* `python src/inventory.py`

---

## 🖥️ Running the Streamlit Dashboard

To launch the interactive multi-page web dashboard, run the following command in your terminal:
```bash
streamlit run app.py
```
This will start a local web server (usually at `http://localhost:8501`) where you can interactively explore the executive summaries, customer segments, risk scores, demand forecasting charts, and inventory actions.
