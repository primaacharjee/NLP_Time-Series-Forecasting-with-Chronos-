# Time Series Forecasting: From Seasonal Trends to Stock Markets
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
- [Methodology](#methodology)  
- [Experiments](#experiments)   
- [Conclusion](#conclusion)   
- [References](#references)  
## Overview
This repository contains the implementation of our study on Time series forecasting applying Chronos  framework. 
This study explores two key aspects of **Chronos**, a pretrained model family for time series forecasting:

1. **Data Enrichment** – Investigating whether incorporating structured features (e.g., weather conditions, geographical attributes) enhances forecasting accuracy.  
2. **Domain Adaptation** – Evaluating how well Chronos generalizes to dynamic environments, such as stock markets influenced by external factors (e.g., energy crises).  

The goal is to assess **Chronos- strengths and limitations** in real-world forecasting applications.

## Methodology

## Data Enrichment and Domain Adaptation for Forecasting

### Data Enrichment for Weather Forecasting
Enhancing forecasting accuracy by incorporating covariates like day, month, season, latitude, longitude, and elevation.

Zero-shot Forecasting: Uses Chronos’ pre-trained models to predict future temperatures directly from covariates.
Forecasting Function: Models the relationship between covariates and temperature to optimize accuracy.

### Domain Adaptation for Stock Price Prediction
Applying time series forecasting to stock prices over a 24-business-day horizon.

Zero-shot Forecasting: Predicts future stock prices using Chronos without training on historical data.
Fine-tuned Forecasting: Improves predictions by training on past stock prices, adapting to company-specific trends.
Both methods refine forecasting across weather and financial domains, improving accuracy through enriched data and model adaptation.

## Experiments
## Time Series Forecasting with Chronos & AutoGluon

## Dataset
### Weather Forecasting
- **Source:** [Kaggle: Philippine Weather Dataset (2020-2023)]([https://www.kaggle.com/](https://www.kaggle.com/datasets/bwandowando/philippine-cities-weather-data-2020-2023))
- **Target Variable:** Mean Air Temperature  
- **Preprocessing:** 10 cities sampled, geospatial features added  
- **Structure:**  
  - `item_id` – City  
  - `timestamp` – Datetime  
  - `target` – Temperature  
  - `covariates` – Day, season, latitude, elevation  

### Stock Price Prediction (Domain Adaptation)
- **Source:** [Kaggle: Energy Crisis & Stock Price Dataset (2021–2024)]([https://www.kaggle.com/](https://www.kaggle.com/datasets/pinuto/energy-crisis-and-stock-price-dataset-2021-2024))  
- **Target Variable:** Close Price  
- **Preprocessing:** Forward fill for missing values  
- **Structure:**  
  - `item_id` – Company (XOM, SHEL, BP)  
  - `timestamp` – Datetime  
  - `target` – Close Price  

## Experimental Design
- **Hardware:** Google Colab (T4 GPU)  
- **Frameworks:** AutoGluon (`autogluon.timeseries`), Chronos  
- **Models:**  
  - Chronos `zeroshot`, `bolt_small` (CAT & XGB Regressor)  
  - Fine-tuned `bolt_small` (2,000 steps, LR = $1e^{-4}$)  
- **Hyperparameters:** Prediction length (24 days), quantile levels (0.1–0.9), WQL/MSE metrics  

## Results & Analysis
### Weather Forecasting
- **Best Model:** ChronosXGBoost (`bolt-base`)  
- **Forecast:** 90-day temperature prediction for 10 cities  

### Stock Market Prediction
- **Separate Training:**  
  - Fine-tuning improved XOM results (-58.7% MSE)  
  - Overfitting in SHEL & BP, fine-tuned models performed worse  
- **Combined Training:**  
  - Zero-shot outperformed fine-tuned models for BP & SHEL  
  - Larger dataset improved generalization  

## Key Findings
- Fine-tuning benefits some stocks but not all  
- Zero-shot models perform well with diverse datasets  
- Generalized models work, but individual training may yield better results  

---
🔗 **For detailed results & plots, please check the full report.**

