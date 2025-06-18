# Project: EUR/USD Direction Prediction Using Classical and Neural Network Models

## Problem Statement

This project investigates the prediction of EUR/USD daily direction (up/down) using both classical and neural network-based machine learning models. The task is framed as a binary classification problem where the target variable indicates whether the closing price for the next day will be higher than today's closing price. The dataset spans over 20 years of historical foreign exchange data and incorporates technical indicators for improved predictive power.

## Dataset

The dataset used is a publicly available historical EUR/USD dataset, containing Open, High, Low, Close (OHLC) prices and Volume. Technical indicators like RSI, MACD, ATR, ADX, moving averages, and volatility were engineered using the `ta` or `talib` Python libraries. Additional features include price change, log-transformed volume, and rolling spreads.

- Source: [DukasCopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
- [Google Sheets CSV Link](https://docs.google.com/spreadsheets/d/1vy592idgw6ifdccWXf1nV6Wf1VGkqcvCoj4OnbU4loI/export?format=csv)

---

## Neural Network Training Instances

The table below summarizes five training instances of neural networks, each using different optimization techniques. Metrics are based on the validation dataset.

| Instance     | Optimizer | Reg/Dropout               | Epochs | Early Stop | Layers           | LR     | Accuracy | Loss   | F1-score | Precision | Recall |
| ------------ | --------- | ------------------------- | ------ | ---------- | ---------------- | ------ | -------- | ------ | -------- | --------- | ------ |
| 1 (Baseline) | SGD       | None                      | 100    | No         | \[64, 32]        | 0.01   | 0.5107   | 0.7453 | 0.4975   | 0.5144    | 0.4817 |
| 2            | Adam      | Dropout(0.3)              | 200    | Yes (5)    | \[128, 64]       | 0.001  | 0.5174   | 0.6956 | 0.4834   | 0.5236    | 0.4489 |
| 3            | RMSprop   | L1 (0.001)                | 150    | Yes (5)    | \[64, 64, 32]    | 0.0005 | 0.5019   | 0.6968 | 0.5205   | 0.5045    | 0.5376 |
| 4            | AdamW     | L1\_L2 (0.01) + Drop(0.4) | 100    | No         | \[256, 128]      | 0.005  | 0.4971   | 0.9310 | 0.1722   | 0.5000    | 0.1040 |
| 5            | Nadam     | L2 (0.001) + Drop(0.5)    | 300    | Yes (10)   | \[512, 256, 128] | 0.0001 | 0.5068   | 0.7347 | 0.5005   | 0.5100    | 0.4913 |

---

## Summary of Findings

**Classical ML Models vs. Neural Networks:**

Our experiments demonstrate that classical ML models and neural networks each have strengths and weaknesses in predicting EUR/USD direction. The tuned Support Vector Machine (SVM) model achieved the highest F1 score (0.6684) on the test set, driven by perfect recall (1.0000) but modest precision (0.5019). This suggests the SVM was highly aggressive in predicting “up” movements, capturing all actual positives but also generating many false positives. In contrast, the best neural network (Instance 3: RMSprop + L1 + Early Stopping) yielded a more balanced trade‑off, with precision at 0.5098, recall at 0.5541, and an overall F1 of 0.5310. Although the NN’s F1 was lower than the SVM’s, its improved precision reduces false alerts, which could be preferable in a trading context that penalizes overtrading.

**Insights from Neural Network Optimization:**

* **Instance 1 (Baseline with SGD):** A basic two‑layer network achieved an F1 of 0.4975. Without any regularization or advanced optimizers, it struggled to generalize, overfitting early and exhibiting high validation loss (0.7453) compared to later instances.

* **Instance 2 (Adam + Dropout):** Introducing dropout at 30% reduced overfitting (validation loss 0.6956), but the F1 fell slightly to 0.4834. Dropout improved robustness but led to underfitting when used alone, evidenced by lower recall (0.4489).

* **Instance 3 (RMSprop + L1 + Early Stopping):** The combination of the RMSprop optimizer, L1 regularization (0.001), and early stopping yielded the best neural network performance. Early stopping prevented overtraining after approximately 20 epochs, while L1 regularization encouraged sparsity. This model balanced precision (0.5045) and recall (0.5376) effectively, resulting in the highest NN F1 (0.5205) on validation and strong test‑set metrics.

* **Instance 4 (AdamW + L1\_L2 + BatchNorm):** Although a sophisticated configuration, the deeper network with dual regularization and batch normalization underperformed (F1 of 0.1722). The high L1\_L2 penalty (0.01) likely constrained the weights excessively, leading to underfitting and a spike in validation loss (0.9310).

* **Instance 5 (Nadam + Deep Architecture):** A deeper three‐layer design with Nadam and strong dropout (50%) returned an F1 of 0.5005. While this model had potential, its complexity required more data and tuning to outperform simpler networks.

**Key Takeaways:**

1. **Optimizer Choice Matters:** RMSprop with L1 and early stopping outperformed Adam and Nadam for this dataset, likely due to its adaptive learning rate handling of sparse signals.
2. **Regularization Balance:** L1 regularization effectively reduced overfitting without collapsing performance; overly aggressive dual regularization hindered learning.
3. **Early Stopping Utility:** Monitoring validation loss prevented wasted epochs and preserved model states at their performance peak.
4. **Model Complexity:** Deeper architectures did not guarantee better results on limited FX data—simplicity with targeted optimizations was more effective.

---

## Instructions to Run

1. **Open the notebook:**

```bash
jupyter notebook Summative_Intro_to_ml_[Abiodun_Kumuyi]_assignment.ipynb
```
2. **Install dependencies:**

Run the cell for installing dependencies

3. **Explore the notebook:**

Run all cells to see outputs.

5. **To load best models:**

```python
# For scikit-learn models
import joblib
svm_model = joblib.load("saved_models/svm.pkl")

# For best neural network model
from tensorflow.keras.models import load_model
nn_model = load_model("saved_models/best_nn_model.keras")
```

---

## File Structure

```
FX_DIRECTION_PREDICTIVE_MODEL/
├── saved_models/
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   ├── best_nn_model.keras
│   ├── nn_instance_1.keras
│   ├── nn_instance_2.keras
│   ├── nn_instance_3.keras
│   ├── nn_instance_4.keras
│   ├── nn_instance_5.keras
├── README.md
├── Summative_Intro_to_ml_[Abiodun_Kumuyi]_assignment.ipynb
├── architecture.png
```

---

## Final Notes

* **Data splits**: 70% training, 15% validation, 15% test
* **Model saving**: All models are saved in `saved_models/`
* **Test Evaluation** was completed for both ML and NN models with confusion matrices and classification reports

## Video

> Video[Include 5-minute video link here – camera must be on, explain optimization choices and result interpretation.]

---
