# Melbourne Housing Price Predictor

A machine learning web application that predicts residential property sold prices across three Melbourne suburbs: **Richmond**, **Hawthorn**, and **Box Hill**.


---

## Live App

[▶ Open the app]([https://melbourne-housing-predictor.streamlit.app](https://melbourne-housing-parsa.streamlit.app/))

---

## Project Overview

This project builds a full ML pipeline from data collection through to a deployed web application:

- **Data collection** — Sold listings scraped from a major Australian real estate platform (Oct 2025 – May 2026), 2,998 records across 3 suburbs
- **Feature engineering** — Schools nearby (Victorian Government open data API), distance to CBD, sale method flags, date features
- **Models** — Linear Regression, Random Forest, XGBoost with 5-fold cross-validation
- **Best model** — XGBoost (R² ≈ 0.87, MAE ≈ $190,000)
- **Deployment** — Streamlit web app
- **LLM comparison** — Claude vs XGBoost on a 20-property sample

---

## Dataset

| Property | Detail |
|---|---|
| Source | Major Australian real estate platform |
| Suburbs | Richmond (VIC 3121), Hawthorn (VIC 3122), Box Hill (VIC 3128) |
| Records | 2,998 sold listings |
| Date range | October 2025 – May 2026 |
| Target variable | Sold price (AUD) |

> **Note:** The raw dataset is not included in this repository. Data was collected for educational purposes only as part of a university assignment.

---

## Features Used

| Feature | Description |
|---|---|
| `suburb` | Richmond / Hawthorn / Box Hill |
| `property_type` | House, Apartment, Townhouse, Villa, Studio, etc. |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `car_spaces` | Number of car spaces |
| `land_size_sqm` | Land size in sqm (0 for apartments) |
| `has_land` | Binary flag — 1 if land size recorded |
| `is_auction` | Binary flag — 1 if sold at auction |
| `month_sold` | Month of sale (1–12) |
| `year_sold` | Year of sale |
| `dist_to_cbd_km` | Straight-line distance to Melbourne CBD |
| `schools_nearby` | Count of open schools within 2km radius |

---

## Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~$350,000 | ~$600,000 | ~0.65 |
| Random Forest | ~$200,000 | ~$400,000 | ~0.85 |
| **XGBoost** | **~$190,000** | **~$380,000** | **~0.87** |

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/melbourne-housing-predictor.git
cd melbourne-housing-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> Requires `xgboost_pipeline.pkl` in the same folder. The model file is included in this repo.

---

## Repository Structure

```
├── app.py                    # Streamlit web application
├── xgboost_pipeline.pkl      # Trained XGBoost pipeline (preprocessor + model)
├── SIT720_Melbourne_Housing_Prediction.ipynb  # Full analysis notebook
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, XGBoost, SHAP
- **Web app** — Streamlit
- **External data** — Victorian Government open data API (school locations)
- **LLM comparison** — Anthropic Claude API

---

## Disclaimer

Data was collected for educational purposes only as part of a University assignment. This project is not intended for commercial use.

---

*Parsa Majidifard Deakin University*
