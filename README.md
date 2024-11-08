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

2. **Data Cleaning**: We removed unnecessary columns such as Age Group, Pregnancy Risk Category, and BMI Category that were not needed for modeling.
   ```r
   data <- data[, !(colnames(data) %in% c("Age Group", "Pregnancy Risk    Category", "BMI Category"))]

3. **Handling Missing Values**: We checked for and handled any missing data.
   ```r
   sum(is.na(Diabetes_Healthcare_Data))

4. **Data Splitting**: We divided the dataset into training (80%) and testing (20%) sets to train and evaluate the model.
   ```r
      set.seed(123)
      train_indices <- sample(1:nrow(Diabetes_Healthcare_Data), 0.8 *         nrow(Diabetes_Healthcare_Data))
   train_data <- Diabetes_Healthcare_Data[train_indices, ]
   test_data <- Diabetes_Healthcare_Data[-train_indices, ]

5. **Exploring Relationships Between Variables**: Goal: Understand how variables like BMI and Plasma Glucose are related to diabetes status, and identify any significant patterns or trends.

In this section, we explore the relationships between key features in the dataset, such as Body Mass Index (BMI), Plasma Glucose levels, and Blood Pressure, and their influence on the target variable: Diabetes Status. Visualizing these relationships helps us understand the potential predictive power of each variable.

2.1 **Plasma Glucose Levels**: We start by examining the distribution of plasma glucose levels in the dataset using a histogram.
   
   # Histogram of Plasma Glucose Levels
      hist(Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
        main="Histogram of Glucose Levels", 
        xlab="Plasma Glucose (mg/dL)", 
        ylab="Frequency", 
        col="lightblue")

2.2 **BMI vs Glucose Levels**: Next, we explore the relationship between Body Mass Index (BMI) and Plasma Glucose levels with a scatter plot.
   # Scatter plot of BMI vs. Plasma Glucose Levels
      plot(Diabetes_Healthcare_Data$`Body Mass Index (kg/m²)`, 
         Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
         main="BMI vs. Plasma Glucose", 
         xlab="Body Mass Index (kg/m²)", 
         ylab="Plasma Glucose (mg/dL)", 
         col="lightblue", 
         pch=19)

2.3 **Blood Pressure vs Diabetes Status**: To investigate whether Blood Pressure varies with Diabetes Status, we use a boxplot to compare the distribution of diastolic blood pressure between diabetic and non-diabetic individuals.

   # Boxplot of Blood Pressure by Diabetes Status
      boxplot(`Diastolic Blood Pressure (mm Hg)` ~ `Diabetes Status (1 =    Positive, 0 = Negative)`, 
        data = Diabetes_Healthcare_Data, 
        main="Blood Pressure by Diabetes Status", 
        xlab="Diabetes Status (1 = Positive, 0 = Negative)", 
        ylab="Blood Pressure (mm Hg)", 
        col="lightblue")

2.4 **Correlation Between Glucose and BMI**: We also explore the correlation between Plasma Glucose levels and BMI to quantify their relationship. A positive correlation suggests that as one variable increases, the other tends to increase as well.
   
   # Correlation between Plasma Glucose and BMI
      cor(Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
       Diabetes_Healthcare_Data$`Body Mass Index (kg/m²)`)







