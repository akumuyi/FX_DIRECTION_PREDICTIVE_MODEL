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

---

## Instructions to Run

1. **Open the notebook:**

```bash
jupyter notebook Summative_Intro_to_ml_[Abiodun_Kumuyi]_assignment.ipynb
```
2. **Install dependencies:**

```text   
Run the cell for installing dependencies
```

4. **Explore the notebook:**

```text
Run all cells to see outputs
```

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

> [Video](https://youtu.be/XQ--iiXhZRg)]

---
