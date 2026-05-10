# 🚗 Car Insurance Claims Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![Model](https://img.shields.io/badge/Best%20Model-Random%20Forest-darkgreen?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-83%25-brightgreen?style=for-the-badge)

---

## 🔍 Problem Statement

Car insurance companies face significant financial risk when customers file unexpected claims.
Accurate prediction of claim likelihood enables smarter premium pricing and better risk management.

> **Goal:** Build a machine learning model that predicts whether a customer will file a car insurance claim —
> enabling insurers to identify high-risk customers **before** losses occur.

---

## 🎯 Target Variable

| Value | Meaning |
|-------|---------|
| `0` | Customer did NOT file a claim ✅ |
| `1` | Customer filed a claim ❌ |

**Claim Rate in Dataset: 31.33%**

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Rows | 10,000 customers |
| Features | 18 input features + 1 target |
| Task | Binary Classification |
| Source | Car Insurance Claim Dataset |

### Key Features

| Feature | Description |
|---------|-------------|
| `DRIVING_EXPERIENCE` | Years of driving (0-9y, 10-19y, 20-29y, 30y+) |
| `VEHICLE_OWNERSHIP` | Owns or leases the vehicle |
| `VEHICLE_YEAR` | Before or after 2015 |
| `AGE` | Driver age group (16-25, 26-39, 40-64, 65+) |
| `INCOME` | Income level (poverty, working class, middle class, upper class) |
| `EDUCATION` | Education level (none, high school, university) |
| `GENDER` | Male or Female |
| `ANNUAL_MILEAGE` | Miles driven per year |
| `CREDIT_SCORE` | Credit score (0-1 normalized) |
| `SPEEDING_VIOLATIONS` | Number of speeding tickets |
| `DUIS` | DUI incidents |
| `PAST_ACCIDENTS` | Previous accidents |

---

## ⚙️ Methodology

```
Data Loading → EDA → Data Cleaning → Feature Analysis
→ Preprocessing → Modeling → Permutation Importance → Explanatory Analysis
```

### Key Steps
- **EDA** — Explored claim rates across all demographic and behavioral features
- **Data Cleaning** — Fixed dtypes for binary columns (MARRIED, CHILDREN, VEHICLE_OWNERSHIP)
- **Feature Analysis** — Identified correlated features (AGE & DRIVING_EXPERIENCE) and low-impact features (RACE, POSTAL_CODE, CREDIT_SCORE)
- **Preprocessing** — StandardScaler for numeric + OneHotEncoder via ColumnTransformer
- **Modeling** — Random Forest Classifier (Default)
- **Permutation Importance** — Unbiased feature ranking
- **Explanatory Analysis** — Deep dive into top 2 predictive features with business-ready visuals

---

## 📈 Visualizations

### Claim Distribution
![Claim Distribution](images/3.png)

### Categorical Features — Claim Rate
![Categorical EDA](images/5.png)

### Numerical Features vs Claim
![Numerical EDA](images/4.png)

### Top 10 Most Important Features — Permutation Importance
![Permutation Importance](images/permutation_importance.png)

### Explanatory Analysis 1 — Driving Experience & Claim Rate
![Driving Experience](images/1.png)

### Explanatory Analysis 2 — Gender & Claim Rate
![Gender](images/2.png)

---

## 🔑 Key EDA Findings

### 🚘 Driving Experience
> Drivers with **0-9 years** of experience have a **62.8%** claim rate —
> nearly **double the average** of 31.3%.
> Expert drivers (30y+) have a claim rate of almost **0%**.

### 👤 Gender
> Male drivers show a **36.3%** claim rate vs **26.4%** for female drivers.
> A **10% gap** — gender is a statistically significant risk predictor.

### 🏎️ Vehicle Year
> Pre-2015 vehicles have a **39.4%** claim rate vs only **10.5%** for newer vehicles.
> Older vehicles = significantly higher risk.

### 📚 Education
> Drivers with **no education** have the highest claim rate at **~45%**.
> University-educated drivers show the lowest risk at **~23%**.

### 💰 Income
> **Poverty-level** income drivers show a **~65%** claim rate.
> Upper class drivers show the lowest at **~27%**.

### 👶 Age
> **16-25 year olds** have a **70%+** claim rate — the highest risk group.
> **65+ drivers** have the lowest claim rate at **~10%**.

---

## 🤖 Model Results

### Random Forest — Default

| Dataset | Precision (0) | Recall (0) | Precision (1) | Recall (1) | Accuracy |
|---------|--------------|------------|--------------|------------|----------|
| Training | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **Test** | **0.86** | **0.90** | **0.76** | **0.67** | **0.83** |

> ⚠️ Training accuracy = 1.00 → Overfitting detected.
> Test Recall for claims (class 1) = 0.67 — room for improvement.

### Most Important Metric
**Recall** — Missing a high-risk customer (False Negative) means
underpricing their premium, which directly costs the company money.

---

## 🔑 Feature Importance — Top 10 (Permutation)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `DRIVING_EXPERIENCE` | 0.062 |
| 2 | `VEHICLE_OWNERSHIP` | 0.053 |
| 3 | `VEHICLE_YEAR` | 0.042 |
| 4 | `POSTAL_CODE` | 0.031 |
| 5 | `GENDER` | 0.028 |
| 6 | `PAST_ACCIDENTS` | 0.024 |
| 7 | `RACE` | 0.008 |
| 8 | `SPEEDING_VIOLATIONS` | 0.007 |
| 9 | `CHILDREN` | 0.006 |
| 10 | `VEHICLE_TYPE` | 0.005 |

### Opportunities for Dimensionality Reduction
- `AGE` & `DRIVING_EXPERIENCE` are highly correlated → candidates for PCA
- `SPEEDING_VIOLATIONS`, `DUIS`, `PAST_ACCIDENTS` capture similar "risky behavior" patterns
- `POSTAL_CODE` functions as an ID — low predictive value despite high permutation importance
- `RACE` showed near-zero difference in claim rates during EDA

---

## 💡 Business Recommendations

### 1. 🎯 Experience-Based Premium Pricing
> Drivers with **less than 10 years** of experience have **2x the average** claim rate.
> Driving experience is the **#1 risk factor** — premiums should reflect this directly.

### 2. 🚗 Flag Pre-2015 Vehicle Owners
> Vehicles manufactured **before 2015** show a **39.4%** claim rate vs **10.5%** for newer vehicles.
> Older vehicles likely lack modern safety features — apply higher risk premiums.

### 3. 👥 Build Behavioral Risk Profiles
> `DRIVING_EXPERIENCE` + `VEHICLE_OWNERSHIP` + `PAST_ACCIDENTS` together
> provide the strongest combined predictive signal.
> A unified behavioral risk score would outperform single-feature pricing.

---

## 🛠️ Tech Stack

```python
pandas • numpy • matplotlib • seaborn • scikit-learn
```

---

## 📁 Project Structure

```
Car-Insurance-Claims-Prediction/
│
├── Projec_4_part_1.ipynb     ← Main notebook
├── README.md
└── images/
    ├── 1.png                 ← Driving Experience vs Claim Rate
    ├── 2.png                 ← Gender vs Claim Rate
    ├── 3.png                 ← Claim Distribution
    ├── 4.png                 ← Numerical Features
    ├── 5.png                 ← Categorical Claim Rates
    └── permutation_importance.png
```

---

## 🚀 How to Run

```bash
git clone https://github.com/dohaalnabahin/Car-Insurance-Claims-Prediction
```

Then open `Projec_4_part_1.ipynb` in Google Colab or Jupyter Notebook.

---

*Built with ❤️ as part of a Data Science learning journey*
