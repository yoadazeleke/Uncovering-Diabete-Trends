# Predictive Modeling for Diabetes Status Prediction

## Project Overview

In this case study, we aim to predict whether an individual has diabetes based on certain features using **Logistic Regression**. This project demonstrates the steps of data preprocessing, feature engineering, model building, and evaluation using a real-world healthcare dataset.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building and Evaluation](#model-building-and-evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Future Improvements](#future-improvements)

## Problem Statement

Diabetes is a growing global health concern, and early detection plays a crucial role in managing its impact. The goal of this project is to develop a machine learning model to predict whether a person has diabetes based on a set of medical and lifestyle factors.

- **Target Variable**: `Diabetes_Status` (1 = Positive, 0 = Negative)
- **Objective**: Build a logistic regression model that predicts whether a person has diabetes based on input features.

## Dataset Description

The dataset used in this case study is a simulated dataset containing several features related to a person's health. These features include demographic and medical information such as:

- **Age**
- **BMI**
- **Blood Pressure**
- **Glucose Levels**
- **Pregnancy History**
- **Family History of Diabetes**

**Target Variable**: `Diabetes_Status`, which indicates whether the person is diabetic (1) or not (0).

## Exploratory Data Analysis (EDA)

### Initial Data Inspection

Before jumping into the modeling process, we first performed some exploratory data analysis (EDA) to understand the structure and contents of the dataset.

1. **Data Inspection**: We examined the first few rows of the dataset and checked column names.
   ```r
   head(data)
   colnames(data)
