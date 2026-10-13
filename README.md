# Flight Delay Prediction & Aviation Analytics Platform

An end-to-end Machine Learning and Aviation Analytics platform built using real **US Bureau of Transportation Statistics (BTS)** flight records (January 2022).

The platform features:
1. **Interactive Flight Delay Predictor**: Real-time binary delay classification (**On-Time** vs. **Delayed** at $\ge 15$ mins), calibrated probability scores, estimated delay duration in minutes, and operational risk factor explanations.
2. **Aviation Analytics Dashboard**: Empirical KPIs across **563,737 flights**, daily disruption timelines (capturing severe holiday disruption peaking on Jan 2 at $43.16\%$ delay rate), 10-carrier benchmarking, top airport hub congestion, and root-cause analysis across the 5 official FAA delay factors.
3. **Model Performance & Intelligence**: Live model comparisons across Logistic Regression, Decision Trees, Random Forests, and LightGBM with confusion matrices and feature importance rankings.

---

## 🛫 Dataset Inventory (`Dataset/`)

The platform is powered exclusively by historical BTS flight records:

- **Total Records**: `563,737` flights
- **Temporal Period**: January 1–31, 2022
- **Completed Flights**: `526,894` ($93.46\%$)
- **Cancelled Flights**: `35,667` ($6.33\%$)
- **Diverted Flights**: `1,176` ($0.21\%$)
- **On-Time Rate**: `80.54%` ($424,379$ flights)
- **Delay Rate ($\ge 15$ min)**: `19.46%` ($102,515$ flights)
- **Airlines (10 Carriers)**: American Airlines (`AA`), Delta Air Lines (`DL`), United Airlines (`UA`), Southwest Airlines (`WN`), Alaska Airlines (`AS`), JetBlue (`B6`), Spirit (`NK`), Frontier (`F9`), Allegiant (`G4`), Hawaiian (`HA`).
- **Airports**: 369 unique US origin hubs (Top: ATL, ORD, DFW, DEN, CLT, LAX, LGA, PHX, SEA, LAS).

---

## 🛡️ Data Hygiene & Leakage Prevention

1. **Isolation of Non-Operational Flights**:
   - Cancelled and diverted flights are isolated for operational status analytics and strictly excluded from delay duration modeling.
2. **Domain-Specific Imputation**:
   - For on-time completed flights, delay cause breakdowns are cleanly set to `0.0` minutes (correcting naive median imputations).
3. **Zero Post-Departure Leakage**:
   - The feature matrix strictly utilizes attributes available *prior* to departure (`Airline`, `Origin`, `Dest`, `CRSDepTime`, `DayOfWeek`, `DayofMonth`, `Distance`, `CRSElapsedTime`).
   - All 30+ post-departure columns (`DepDelay`, `ArrDelay`, `Taxi-out`, `Wheels Off/On`, `AirTime`, delay causes) are barred from prediction.

---

## 🤖 Machine Learning Model Architecture

### Dual-Stage Inference Pipeline
- **Stage 1 (Binary Classifier)**: Predicts probability of delay ($P(\text{Delayed})$) against the FAA standard threshold of $\ge 15$ minutes.
- **Stage 2 (Duration Regressor)**: When a flight is classified as delayed, a secondary regressor estimates the expected delay duration in minutes.

### Candidate Model Comparison (Test Set: 28,561 Flights)

| Candidate Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | $63.00\%$ | $27.68\%$ | $61.84\%$ | $0.3824$ | $0.6723$ | $0.3325$ | Evaluated |
| **Decision Tree** | $68.17\%$ | $31.49\%$ | $61.07\%$ | $0.4155$ | $0.7009$ | $0.3851$ | Evaluated |
| **Random Forest** | $69.75\%$ | $33.07\%$ | $61.80\%$ | $0.4308$ | $0.7232$ | $0.4128$ | Evaluated |
| **LightGBM** | **70.39%** | **34.20%** | **64.73%** | **0.4475** | **0.7494** | **0.4482** | **Champion** |

- **Delay Regressor Test Performance**: MAE = $46.08\text{ minutes}$ | RMSE = $85.29\text{ minutes}$.

---

## 🚀 Quickstart & Setup

### 1. Environment Installation

#### 🪟 Windows (PowerShell / Command Prompt / Git Bash)

**Using PowerShell:**
```powershell
# 1. Clone or navigate to the repository directory
cd "Flight Delay Prediction and Aviation Analytics"

# 2. Create virtual environment
python -m venv .venv

# 3. Allow script execution in current PowerShell session (if blocked)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 4. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 5. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Using Command Prompt (cmd.exe):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Using Git Bash on Windows:**
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

---

#### 🐧 Linux / macOS (Bash / Zsh)

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Run Automated Test Suite

On any platform (Windows / Linux / macOS):
```bash
pytest tests/ -v
# or with python module launcher:
python -m pytest tests/ -v
```
*(All 22 unit and integration tests pass).*

---

### 3. Launch Streamlit Application

You can launch using the root launcher or direct app path:

```bash
# Recommended root launcher
streamlit run streamlit_app.py

# Or via python module (recommended if 'streamlit' command is not in system PATH on Windows):
python -m streamlit run streamlit_app.py
```

---

### 4. Retrain Models (Optional)

```bash
python -m src.train
```

---

## 📂 Project Structure

```text
Flight Delay Prediction and Aviation Analytics/
├── Dataset/
│   ├── Flight Delay Original Dataset.csv
│   ├── Flight Delay Selected Dataset.csv
│   └── Flight Delay Cleaned Dataset.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Optimized chunked/cached CSV reader
│   ├── cleaning.py           # Domain cleaning & leakage prevention guard
│   ├── feature_engineering.py# Pre-flight derivations & time block bins
│   ├── pipeline.py           # Scikit-Learn ColumnTransformer pipeline
│   ├── train.py              # Candidate training & champion selection
│   ├── predict.py            # Standalone inference API with delay minutes
│   └── analytics.py          # KPI engine & aggregations
├── models/
│   ├── flight_delay_classifier.pkl
│   ├── flight_delay_regressor.pkl
│   ├── preprocessor_pipeline.pkl
│   └── model_metadata.json
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py      # Main entry point with 3-tab layout
│   ├── components_predict.py # Flight delay prediction form & result UI
│   └── components_analytics.py # KPI cards, timelines, and Plotly charts
├── tests/
│   ├── test_data_loader.py
│   ├── test_cleaning.py
│   ├── test_pipeline.py
│   ├── test_prediction.py
│   ├── test_analytics.py
│   └── test_integration.py
├── requirements.txt
├── workflow.md
└── README.md
```
