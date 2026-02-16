# Rhea Soil Nutrient Prediction Challenge

This repository contains the solution for the [Rhea Soil Nutrient Prediction Challenge](https://zindi.africa/competitions/rhea-soil-nutrient-prediction-challenge) hosted on Zindi.

## 📌 Project Overview

Soil health is the cornerstone of sustainable agriculture. In many parts of Africa, traditional laboratory soil testing is expensive, slow, and inaccessible to smallholder farmers.

The goal of this challenge is to develop a machine learning model that estimates the levels of **13 essential soil nutrients** using a combination of geospatial and existing soil data. By predicting these nutrients at locations where lab tests aren't available, we help **Rhea** provide tailored recommendations to farmers, ensuring better yields and long-term sustainability.

## 🎯 Objectives

* **Predict 13 target nutrients:** Aluminum (Al), Boron (B), Calcium (Ca), Copper (Cu), Iron (Fe), Magnesium (Mg), Manganese (Mn), Phosphorus (P), Potassium (K), Sodium (Na), Sulfur (S), Zinc (Zn), and Nitrogen (N).
* **Minimize RMSE:** The primary evaluation metric for this regression task is the Root Mean Squared Error (RMSE).

## 📊 Dataset Description

The dataset consists of geospatial features and soil nutrient data:

* **Training Data:** Locations with lab-tested results for the 13 target nutrients.
* **Test Data:** Locations where nutrient levels need to be predicted.
* **Features:** Environmental variables, geospatial coordinates, and potentially satellite-derived indicators (depending on the specific Zindi data provided).

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost/LightGBM/CatBoost, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/samyakrajbayar/Rhea-Soil-Nutrient-Prediction-Challenge.git
cd Rhea-Soil-Nutrient-Prediction-Challenge

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Data Preparation

Place the competition data files (`train.csv`, `test.csv`, `SampleSubmission.csv`) in the `data/` folder.

### 4. Run the Analysis

Open the Jupyter notebook to see the exploratory data analysis (EDA), feature engineering, and model training:

```bash
jupyter notebook Rhea_Soil_Prediction.ipynb

```

## 📈 Methodology

1. **Exploratory Data Analysis (EDA):** Understanding the distribution of nutrients and correlations between geospatial features.
2. **Feature Engineering:** Creating spatial features, handling missing values, and scaling.
3. **Modeling:** * Baseline models (Random Forest, Linear Regression).
* Gradient Boosting Regressors (XGBoost, LightGBM).
* Multi-output regression strategies to predict all 13 nutrients simultaneously or individually.


4. **Evaluation:** Cross-validation using RMSE to ensure model robustness.

## 📝 Submission Format

The final submission should be a CSV file with an `ID` column (identifying the location) and columns for each of the 13 nutrients (e.g., `Target_Al`, `Target_B`, etc.).


## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE) - see the https://www.google.com/search?q=LICENSE file for details. (Note: Ensure your code adheres to the competition's open-source requirements).

## 🙏 Acknowledgments

* [Zindi Africa](https://zindi.africa) for hosting the challenge.
* [Rhea](https://www.rhea.africa/) for providing the data and mission.
* Digital Africa for supporting innovation in the agricultural sector.
