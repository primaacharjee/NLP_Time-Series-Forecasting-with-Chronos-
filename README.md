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

## Features - TBA
- Implements **Retrieval-Augmented Generation (RAG)** for classification.
- Introduces **Corpus Lexical Entropy** as a diversity metric.
- Uses **FAISS** for fast retrieval of similar cases.
- Employs **SBERT-NLI** embeddings for document representation.
- Integrates **Mistral7B** for classification.
- Proposes **[+]EXPL**, a downsampling strategy to mitigate class imbalance. 


## Methodology

#### Extension 1-
#### Extension 2-

## Experiments

#### Experiment 1- Data Enrichment (Weather Forecasting)

### Dataset  
- **Source**: Kaggle – *Philippine Weather Dataset (2020–2023)*  
- **File Used**: `dailydata_combined.csv` (no missing values)  

### Key Features  
- **City** – Location identifier  
- **Datetime** – Time reference  
- **Mean Air Temperature** – Target variable  
- **Covariates**:  
  - Day of the week  
  - Weekend indicator  
  - Season (numerically encoded)   

### Dataset Overview  
- **Total Records**: 206,001  
- **Total Cities**: 137 (Subset: 10 cities)  

### Forecasting Data Structure (Chronos & AutoGluon Compatible)  
| Column      | Description                   |  
|------------|--------------------------------|  
| `item_id`  | City identifier               |  
| `timestamp` | Datetime                      |  
| `target`   | Mean Temperature               |  
| `covariates` | Day of the week, season, latitude, elevation, etc. |  


#### Experiment 2: Domain Adaptation (Stock Market Forecasting)

### Dataset  
- **Stock Price Data**: XOM, SHEL, BP (2021–2024)  

### Models Evaluated  
- **Chronos Zero-Shot** – Pretrained model without fine-tuning  
- **Chronos Fine-Tuned** – Model fine-tuned on stock price data  
- **Chronos Ensemble** – Combination of multiple models for improved robustness  

### Training Configuration  
- **Prediction Length**: 24 business days  
- **Hyperparameters**:  
  - Learning rate: `1e-4`  
  - Training steps: `2000`  

