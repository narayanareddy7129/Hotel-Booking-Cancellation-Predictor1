# 🏨 Hotel Booking Cancellation Prediction

> A full end-to-end Machine Learning project that predicts whether a hotel booking will be cancelled — built using a structured **4-Sprint Agile methodology** reflecting real-world industry workflows.



## Problem Statement

Hotel cancellations are a major revenue problem. A hotel that can predict which bookings are likely to be cancelled can take proactive steps — targeted outreach, overbooking strategies, dynamic pricing — to minimize losses.

This project builds a production-ready ML system that takes a raw booking record and outputs a **cancellation probability**, deployable as an interactive Streamlit web app.

---

## Key Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~86% |
| **F1 Score** | ~85% |
| **ROC-AUC** | ~0.92 |
| **Model** | Stacking Classifier (RF + DT + tuned RF via Optuna) |

---

## Project Structure

```
Hotel-Booking-Cancellation-Prediction/
│
├── data/
│   └── hotel_bookings.csv              # Original Kaggle dataset
│
├── notebooks/
│   ├── Sprint_1_Data_Understanding_and_Preprocessing.ipynb
│   ├── Sprint_2_Model_Building_and_Evaluation.ipynb
│   ├── Sprint_3_Optimization_and_Final_Model.ipynb
│   └── Sprint_4_Deployment.ipynb
│
├── src/
│   └── feature_engineering.py          # Custom sklearn transformers:
│                                        #   RawPreprocessor, FeatureEngineer,
│                                        #   FeatureSelector, ToDataFrame
│
├── models/
│   ├── preprocessor.pkl                # Fitted ColumnTransformer (Sprint 1)
│   ├── final_model.pkl                 # Tuned Stacking Classifier (Sprint 3)
│   ├── full_pipeline.pkl               # End-to-end pipeline (Sprint 4)
│   ├── selected_features.json          # Top-30 feature names
│   ├── adr_upper_bound.json            # ADR outlier cap value
│   └── experiment_log.json             # Experiment tracking log
│
├── app/
│   └── app.py                          # Streamlit web application
│
├── logs/
│   └── predictions.csv                 # Prediction audit log
│
├── requirements.txt
└── README.md
```

---

## 4-Sprint Methodology

### Sprint 1 — Data Understanding & Preprocessing
- Loaded and inspected the **Hotel Booking Demand** dataset (119,000+ bookings, 32 features)
- Handled missing values, removed duplicates, fixed data types
- Performed **Univariate, Bivariate, and Multivariate EDA** with visualisations
- Detected and treated outliers (IQR-based capping for `adr`, log-transform for `lead_time`)
- Applied **ColumnTransformer**: OneHotEncoding for categoricals, StandardScaler for numericals
- Saved `preprocessor.pkl` and `preprocessed_df.csv`

### Sprint 2 — Model Building & Evaluation
- Trained **6 classifiers** as baseline: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, ExtraTrees
- Evaluated each on Accuracy, Precision, Recall, F1, ROC-AUC
- Plotted confusion matrices and ROC curves for all models
- Checked for overfitting via Train vs Test accuracy comparison
- Identified **Random Forest** as the best standalone baseline

### Sprint 3 — Optimization & Final Model
- Engineered **4 new domain features**:
  - `fe__total_nights` — weekday + weekend stays
  - `fe__cancel_ratio` — ratio of previous cancellations to total bookings
  - `fe__lead_x_cancel` — lead time × cancellation history (interaction term)
  - `fe__was_waitlisted` — binary: was the customer on a waiting list?
- Selected **top-30 features** via Random Forest importance
- Tuned hyperparameters with **Optuna** (20 trials, 5-fold CV on F1 score)
- Built a **Stacking Classifier** (tuned RF + Decision Tree + ExtraTrees as base learners, Logistic Regression as meta-learner)
- Saved `final_model.pkl` and `selected_features.json`

### Sprint 4 — Deployment
- Built an **end-to-end sklearn Pipeline**:
  ```
  Raw Data → RawPreprocessor → ColumnTransformer → ToDataFrame
           → FeatureEngineer → FeatureSelector → Stacking Classifier
  ```
- Built an interactive **Streamlit app** with sidebar inputs for all booking fields
- Organised project structure and documented with this README

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction
```

### 2. Create a virtual environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
---

## Running the Notebooks

Run the four Sprint notebooks **in order** from the project root. Each notebook saves artifacts that the next one depends on:

| Order | Notebook | Saves |
|-------|----------|-------|
| 1st | `Sprint_1_Data_Understanding_and_Preprocessing.ipynb` | `preprocessed_df.csv`, `models/preprocessor.pkl` |
| 2nd | `Sprint_2_Model_Building_and_Evaluation.ipynb` | Model comparison results |
| 3rd | `Sprint_3_Optimization_and_Final_Model.ipynb` | `models/final_model.pkl`, `models/selected_features.json`, `src/feature_engineering.py` |
| 4th | `Sprint_4_Deployment.ipynb` | `models/full_pipeline.pkl`, `app/app.py` |

---

## Running the Streamlit App

From the **project root** (not from inside `app/`):

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`. Use the sidebar to enter booking details and click **Predict**.

![App Screenshot](https://github.com/narayanareddy7129/Hotel-Booking-Cancellation-Predictor1/blob/2442046d0966db68e21c87fa3aae38e35665913f/app%20interface%20screenshot.png)

---

## Project Dependencies

Key libraries used:

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.9.0 | ML models, pipelines, preprocessing |
| pandas | 3.0.3 | Data manipulation |
| numpy | 2.4.6 | Numerical computation |
| streamlit | 1.58.0 | Web app frontend |
| optuna | — | Hyperparameter tuning |
| joblib | 1.5.3 | Model serialization |
| matplotlib | 3.11.0 | Visualisation |
| seaborn | 0.13.2 | Statistical plots |

Full list in `requirements.txt`.

---

## Key Findings

- **Lead time** is the strongest single predictor of cancellation — bookings made far in advance cancel much more frequently
- Customers with a **history of previous cancellations** (`cancel_ratio`) are far more likely to cancel again
- The engineered **`fe__lead_x_cancel`** interaction feature (lead time × cancel ratio) ranked among the top-5 features
- **Online TA** (Travel Agency) bookings have the highest cancellation rate by market segment
- **Repeated guests** almost never cancel — a strong negative signal
- Customers on the **waiting list** have a higher cancellation rate once confirmed

---

## Custom Transformer Classes

All custom sklearn-compatible transformers live in `src/feature_engineering.py`:

| Class | Purpose |
|-------|---------|
| `RawPreprocessor` | Converts raw CSV columns → what the ColumnTransformer expects (`lead_time_log`, `adr_capped`, column selection) |
| `ToDataFrame` | Converts ColumnTransformer numpy output → pandas DataFrame with named/deduped columns |
| `FeatureEngineer` | Adds the 4 `fe__` domain features |
| `FeatureSelector` | Selects only the top-30 features by name |

These are imported in `app.py` before `joblib.load()` so that `full_pipeline.pkl` can unpickle correctly.

---

## Author

**Muvva Lakshmi Narayana Reddy**
- [LinkedIn](https://www.linkedin.com/in/<muvva-lakshmi-narayana-reddy-5933a3299>)
- [GitHub](https://github.com/<[your-username](narayanareddy7129/Hotel-Booking-Cancellation-Predictor1/blob/2442046d0966db68e21c87fa3aae38e35665913f/app%20interface%20screenshot.png)>)

---

## Acknowledgements

- Dataset: [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
