# NIFTY IV Surface Reconstruction Challenge

Finance Club IIT Roorkee Open Project 2026

## Overview

This project focuses on reconstructing missing implied volatility (IV) values in a partially observed NIFTY options dataset.

The challenge was evaluated using Mean Squared Error (MSE), with the final ranking determined primarily by the private leaderboard.

---

## Financial Motivation

Implied volatility smiles are more naturally represented using log-moneyness:

```text
k = log(K / S)
```

where:

- K = strike price
- S = underlying price

This representation normalizes strikes relative to spot and is widely used in quantitative volatility modeling.

---

## Methodology

For each timestamp:

1. Convert strikes to log-moneyness.
2. Fit:
   - PCHIP Interpolator
   - CubicSpline Interpolator
3. Predict missing IV values.
4. Ensemble predictions:

```text
IV_pred =
0.5 × IV_PCHIP +
0.5 × IV_CubicSpline
```

---

## Repository Structure

```text
.
├── notebooks/
├── src/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/submission.py
```

This generates:

```text
submission.csv
```

which reproduces the final Kaggle submission.

---

## Tech Stack

- Python
- NumPy
- Pandas
- SciPy
- Jupyter

---
