# New Particle Formation (NPF) Classification

## 📌 Project Overview

This project focuses on **classifying New Particle Formation (NPF) events** using daily atmospheric measurements from the **SMEAR II station (Hyytiälä, Finland)**.

We build a machine learning pipeline to:
- Predict the **daily NPF event type** (`class4 ∈ {Ia, Ib, II, nonevent}`)
- Estimate a **well-calibrated probability** for whether an NPF event occurred (`class2 = event vs nonevent`)

The solution was evaluated in a **Kaggle competition**, where performance depended not only on accuracy but also on **probability quality**.

---

## 🎯 Objectives & Evaluation Metrics

The Kaggle leaderboard score combines three metrics:

1. **Binary Accuracy (class2)**  
   Correct prediction of event vs nonevent.

2. **Perplexity (class2)**  
   Measures the quality of predicted probabilities for event/nonevent  
   → *Lower is better; strongly depends on probability calibration.*

3. **Multiclass Accuracy (class4)**  
   Exact match accuracy for NPF subtypes.

⚠️ **Key design challenge:**  
Optimizing accuracy alone degrades perplexity. This project explicitly optimizes **both classification and probability calibration**.

---

## 🧪 Dataset

- Daily aggregated environmental measurements (means & standard deviations)
- Data source: **SMEAR II research station**
- ~450 training samples
- Highly correlated features across heights
- Balanced training distribution for event vs nonevent

---

## 🔍 Exploratory Data Analysis (EDA)

Key steps:
- Target distribution analysis (binary & multiclass)
- Correlation heatmaps to detect multicollinearity
- PCA & KMeans clustering for unsupervised structure inspection
- Feature redundancy detection (`.mean` / `.std` pairs)

Decisions from EDA:
- Remove constant and redundant features
- Introduce **coefficient-of-variation features**
- Prefer physically interpretable features

---

## 🛠 Preprocessing & Feature Engineering

Main steps:
1. Binary target creation (`class2`)
2. Label encoding for multiclass target (`class4`)
3. Median imputation (robustness)
4. Feature scaling (for linear models only)
5. Feature engineering:
   - Coefficient of variation (`std / |mean|`)
   - Global aggregates of `.mean` and `.std`
6. Feature selection:
   - Variance thresholding
   - Correlation pruning
   - Permutation importance (tree-based models)

---

## 🤖 Models Used

### Baseline Models
- Logistic Regression
- Random Forest

### Final Base Learners
- **ExtraTreesClassifier**
- **XGBoost**
- **RandomForestClassifier**

These models were selected for:
- Strong performance on tabular data
- Native probabilistic outputs
- Diversity for ensembling

---

## 🔁 Cross-Validation & OOF Strategy

- **Stratified 5-Fold Cross-Validation**
- Out-of-Fold (OOF) predictions used for:
  - Fair ensemble weight tuning
  - Probability calibration
  - Preventing data leakage

---

## 🎛 Hyperparameter Tuning

- **Optuna** used for automated tuning
- Each model optimized with ~150 trials
- **Composite objective**:
  ```
  Composite Loss = α · Binary Log Loss + (1 − α) · Multiclass Log Loss
  ```
Where `α = 0.7` prioritizes probability quality (perplexity).

- All experiments logged with **MLflow**

---

## 🧩 Ensembling

- Weighted average of multiclass probabilities from base models
- Ensemble weights tuned using Optuna on OOF predictions
- Objective aligned exactly with leaderboard metrics

---

## 📐 Probability Calibration

- Applied **only to binary probability** (`p_event`)
- Method: **Platt Scaling (Logistic Regression)**
- Calibrated:
```
p_event = 1 − P(nonevent)
```
- This improves perplexity without harming multiclass accuracy

---

## 📊 Results

| Model | Class4 Accuracy | Class2 Accuracy | Perplexity | Aggregated Score |
|------|----------------|----------------|------------|------------------|
| ExtraTrees | 0.660 | 0.867 | 1.402 | 0.708 |
| XGBoost | 0.684 | 0.867 | 1.376 | 0.719 |
| RandomForest | 0.633 | 0.860 | 1.411 | 0.694 |
| **Ensemble** | 0.678 | 0.869 | 1.391 | 0.724 |
| **Ensemble (Calibrated)** | **0.678** | **0.871** | **1.376** | **0.725** |

🏆 **Kaggle Performance**
- Private Leaderboard: **0.74205 (Rank 15/150)**

---

## 💡 Discussion

### Strengths
- Explicit focus on probability calibration
- Leakage-free OOF-based ensembling
- Composite objective aligned with evaluation
- Model diversity improves robustness

### Limitations
- Small dataset (~450 samples)
- Sparse subclasses (Ia, Ib)
- Class4 probabilities not fully calibrated (by design)

---

## 🧰 Tools & Libraries

- Python
- scikit-learn
- XGBoost
- Optuna
- MLflow
- NumPy / Pandas / Matplotlib

---

## 📚 References

Key references include:
- *An Introduction to Statistical Learning* (James et al.)
- XGBoost (Chen & Guestrin, 2016)
- scikit-learn documentation
- SMEAR II research resources
- Atmospheric NPF literature

---

## 🏁 Final Notes

This project demonstrates a **production-style ML workflow** with:
- Robust evaluation
- Probability-aware modeling
- Reproducible experimentation
- Strong leaderboard performance

