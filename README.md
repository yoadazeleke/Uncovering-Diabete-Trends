# Predictive Modeling for Diabetes Status Prediction

## Project Overview

This case study demonstrates the process of predicting diabetes status using **Logistic Regression**. The goal is to predict whether an individual has diabetes based on various health-related features. This project covers data preprocessing, feature engineering, model building, and evaluation using a real-world healthcare dataset.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building and Evaluation](#model-building-and-evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Future Improvements](#future-improvements)
9. [Interactive Dashboard](#interactive-dashboard)

## Problem Statement

Diabetes is a major global health issue, and early detection is crucial in managing its effects. The objective of this project is to develop a machine learning model that can predict whether an individual has diabetes based on medical and lifestyle factors.

- **Target Variable**: `Diabetes_Status` (1 = Positive, 0 = Negative)
- **Objective**: Build a logistic regression model to predict diabetes status based on input features.

## Dataset Description

The dataset for this case study comes from the [Healthcare Diabetes dataset on Kaggle](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes), which holds a quality score of **9.41**. It was provided by the **National Institute of Diabetes and Digestive and Kidney Diseases**.

**Features**:
- **Age**
- **BMI**
- **Blood Pressure**
- **Glucose Levels**
- **Pregnancy History**
- **Family History of Diabetes**

**Target Variable**: `Diabetes_Status`, which indicates whether a person is diabetic (1) or not (0).

## Exploratory Data Analysis (EDA)

1. Initial Data Inspection

To better understand the dataset, we first conducted an exploratory data analysis (EDA), examining the first few rows and column names.

          head(Diabetes_Healthcare_Data)
          colnames(Diabetes_Healthcare_Data)

2. Data CLeaning

We removed irrelevant columns such as "Age Group," "Pregnancy Risk Category," and "BMI Category," which were unnecessary for the prediction task.

          Diabetes_Healthcare_Data <- Diabetes_Healthcare_Data[, !(colnames(Diabetes_Healthcare_Data) %in% c("Age Group", "Pregnancy Risk Category", "BMI Category"))]

3. Handling Missing Data
Next, we checked for missing values and handled them appropriately by imputing or removing rows/columns based on the severity of missing data.

          sum(is.na(Diabetes_Healthcare_Data))


4. Data Splitting
We split the dataset into training (80%) and testing (20%) sets to ensure the model could generalize well to new, unseen data.

          set.seed(123)
          train_indices <- sample(1:nrow(Diabetes_Healthcare_Data), 0.8 * nrow(Diabetes_Healthcare_Data))
          train_data <- Diabetes_Healthcare_Data[train_indices, ]
          test_data <- Diabetes_Healthcare_Data[-train_indices, ]

5. Exploring Relationships Between Variables
We visualized relationships between key features and the target variable to identify useful predictors.

5.1 Plasma Glucose Levels Distribution
We analyzed the distribution of plasma glucose levels using a histogram.

     hist(Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
          main="Histogram of Glucose Levels", 
          xlab="Plasma Glucose (mg/dL)", 
          ylab="Frequency", 
          col="lightblue")
          
5.2 BMI vs. Plasma Glucose
A scatter plot was created to examine the relationship between BMI and plasma glucose levels.

     plot(Diabetes_Healthcare_Data$`Body Mass Index (kg/m²)`, 
          Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
          main="BMI vs. Plasma Glucose", 
          xlab="Body Mass Index (kg/m²)", 
          ylab="Plasma Glucose (mg/dL)", 
          col="lightblue", 
          pch=19)
     
5.3 Blood Pressure vs. Diabetes Status
A boxplot was created to visualize the relationship between blood pressure and diabetes status.

     boxplot(`Diastolic Blood Pressure (mm Hg)` ~ `Diabetes Status (1 = Positive, 0 = Negative)`, 
        data = Diabetes_Healthcare_Data, 
        main="Blood Pressure by Diabetes Status", 
        xlab="Diabetes Status (1 = Positive, 0 = Negative)", 
        ylab="Blood Pressure (mm Hg)", 
        col="lightblue")
        
5.4 Correlation Between Plasma Glucose and BMI
We computed the correlation coefficient between plasma glucose levels and BMI to identify any linear relationship.

     cor(Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
     Diabetes_Healthcare_Data$`Body Mass Index (kg/m²)`)

The correlation coefficient was found to be 0.225, indicating a weak positive correlation.


## Data Preprocessing

In this section, we'll prepare the data for building the model. This involves a few steps to clean the data, handle any missing values, and prepare it for use in the model.

1. Loading the Data
First, we load the data from a CSV file.

          data <- read.csv("Diabetes_Healthcare_Data")

This command reads the data file and stores it in the variable data.

2. Handling Missing Data
Next, we check for and remove any rows that have missing values (NA). This is important because most models can't work with incomplete data.

          data <- na.omit(data)  # Removing rows with NA values

This command removes any row with missing values from the dataset.

3. Encoding Categorical Variables
If our dataset has categorical data (like "Yes" or "No"), we need to convert it into a format the model can understand. In this case, we’re encoding the target variable Diabetes_Status into two categories: Negative and Positive.

          data$Diabetes_Status <- factor(data$Diabetes_Status, levels = c(0, 1), labels = c("Negative", "Positive"))

This code converts the Diabetes_Status column to a factor (categorical variable) with labels "Negative" for 0 and "Positive" for 1.

4. Feature Scaling
Feature scaling is important when we have features (like BMI) with different units of measurement. Scaling helps the model treat each feature equally. In this case, we are scaling the BMI feature.

          data$BMI <- scale(data$BMI)  # Scaling BMI feature

This code standardizes the BMI values, so they all lie within a similar range, making it easier for the model to process.

5. Data Splitting
Now, we split the data into two parts: one for training the model and the other for testing it. We’ll use 80% of the data for training and 20% for testing.

          set.seed(123)
          train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
          train_data <- data[train_indices, ]
          test_data <- data[-train_indices, ]

This code randomly splits the dataset into train_data (80%) and test_data (20%). The set.seed(123) ensures the split is the same every time we run the code.

