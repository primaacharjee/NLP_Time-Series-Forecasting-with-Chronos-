# Heading - TBA
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-1.3.3-blue.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-blue.svg)
![AutoGluon](https://img.shields.io/badge/AutoGluon-0.4.0-blue.svg)  <!-- You may need to adjust the version -->
![Torch](https://img.shields.io/badge/torch-1.11.0-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21.0-blue.svg)
![Kaggle Data](https://img.shields.io/badge/Kaggle-Datasets-blue.svg)
![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellowgreen.svg)


## Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Methodology](#methodology)  
- [Experiments](#experiments)   
- [Conclusion](#conclusion)   
- [References](#references)  
## Overview
This repository contains the implementation of our study on Time series forecasting applying Chronos  framework. 
This study explores two key aspects of **Chronos**, a pretrained model family for time series forecasting:

1. **Data Enrichment** – Investigating whether incorporating structured features (e.g., weather conditions, geographical attributes) enhances forecasting accuracy.  
2. **Domain Adaptation** – Evaluating how well Chronos generalizes to dynamic environments, such as stock markets influenced by external factors (e.g., energy crises).  

The goal is to assess **Chronos' strengths and limitations** in real-world forecasting applications.

## Methodology

## Data Enrichment and Domain Adaptation for Forecasting

### Extension 1: Data Enrichment

Enhance forecasting accuracy using contextual covariates with the following variables:
- $y_t$: Target variable (mean air temperature).
- $X_t$: Covariates including day, month, season, latitude, longitude, elevation.
- $T$: Total time steps.
- $h$: Prediction length (90 days).

**Zero-shot Forecasting:**
\[ \hat{y}_{t+h} = \text{Chronos}_{\text{zero-shot}}(X_t) \]

**Forecasting Function:**
\[ \hat{y}_{t+h} = f(X_t, \theta) \]

**Weighted Quantile Loss (WQL):**
\[ \text{WQL}_q = \frac{1}{N} \sum_{i=1}^{N} w_i \left[ q \cdot \mathbb{I}(y_i \geq \hat{y}_i) + (1-q) \cdot \mathbb{I}(y_i < \hat{y}_i) \right] \]

### Extension 2: Domain Adaptation

Forecast stock prices with historical data:
- $y_t$: Stock price.
- $X_t$: Feature set.
- $T$: Total time steps.
- $h$: Prediction length (24 business days).

**Zero-shot Forecasting:**
$$
\[ \hat{y}_{t+h} = \text{Chronos}_{\text{zero-shot}}(X_t) \]
$$
**Fine-tuned Forecasting:**
$$
\[ \hat{y}_{t+h} = \text{Chronos}_{\text{fine-tuned}}(X_t, \theta) \]
$$
**Mean Squared Error (MSE):**
$$
\[ MSE = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_{t_i+h} - y_{t_i+h} \right)^2 \]
$$
## Experiments

#### Experiment 1- Data Enrichment (Weather Forecasting)

### Dataset  
- **Source**: Kaggle – *Philippine Weather Dataset (2020–2023)*
  
### Dataset Overview  
- **Total Records**: 206,001  
- **Total Cities**: 137 (Subset: 10 cities)  

### Models Evaluated  and Configuration
- **Chronos (bolt_small, bolt_base)**
- **Regressor Types:** `"CAT" (CatBoost), "XGB" (XGBoost)`
- **Target Scalers** : `Standard, Robust, Mean Absolute and, Min-Max`
- **Prediction Length**: `90 days`


#### Experiment 2: Domain Adaptation (Stock Market Forecasting)

### Dataset  
- **Source**: Kaggle- *Stock Price Data: XOM, SHEL, BP (2021–2024)*

### Models Evaluated  
- **Chronos Zero-Shot**  
- **Chronos Fine-Tuned** – Model fine-tuned on stock price data  
- **Chronos Ensemble** – Combination of multiple models for improved robustness  

### Training Configuration  
- **Prediction Length**: 24 business days  
- **Hyperparameters**:  
  - Learning rate: `1e-4`  
  - Training steps: `2000`
  
#### Evaluation Metrics for both experiments
**WQL**- to measure the quality of the probabilistic nature.
It is calculated as follows:

**MSE**- penalizes larger errors, helping Chronos to learn accurate predictions by emphasizing larger discrepancies between predicted and actual values.

#### Results-

