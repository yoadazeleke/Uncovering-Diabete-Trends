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

Features:
- **Age**
- **BMI**
- **Blood Pressure**
- **Glucose Levels**
- **Pregnancy History**
- **Family History of Diabetes**

**Target Variable**: `Diabetes_Status`, which indicates whether a person is diabetic (1) or not (0).

## Exploratory Data Analysis (EDA)

### 1. Initial Data Inspection

To better understand the dataset, we first conducted an exploratory data analysis (EDA), examining the first few rows and column names.

     head(Diabetes_Healthcare_Data)
     colnames(Diabetes_Healthcare_Data)

## 2. Data Cleaning
We removed irrelevant columns such as "Age Group," "Pregnancy Risk Category," and "BMI Category," which were unnecessary for the prediction task.

     Diabetes_Healthcare_Data <- Diabetes_Healthcare_Data[, !(colnames(Diabetes_Healthcare_Data) %in% c("Age Group", "Pregnancy Risk Category", "BMI Category"))]

## 3. Handling Missing Data
Next, we checked for missing values and handled them appropriately by imputing or removing rows/columns based on the severity of missing data.

     sum(is.na(Diabetes_Healthcare_Data))

## 4. Data Splitting
We split the dataset into training (80%) and testing (20%) sets to ensure the model could generalize well to new, unseen data.

     set.seed(123)
     train_indices <- sample(1:nrow(Diabetes_Healthcare_Data), 0.8 * nrow(Diabetes_Healthcare_Data))
     train_data <- Diabetes_Healthcare_Data[train_indices, ]
     test_data <- Diabetes_Healthcare_Data[-train_indices, ]

## 5. Exploring Relationships Between Variables
We visualized relationships between key features and the target variable to identify useful predictors.

## 5.1 Plasma Glucose Levels Distribution
We analyzed the distribution of plasma glucose levels using a histogram.

     hist(Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
          main="Histogram of Glucose Levels", 
          xlab="Plasma Glucose (mg/dL)", 
          ylab="Frequency", 
          col="lightblue")
## 5.2 BMI vs. Plasma Glucose
A scatter plot was created to examine the relationship between BMI and plasma glucose levels.

     plot(Diabetes_Healthcare_Data$`Body Mass Index (kg/m²)`, 
          Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
          main="BMI vs. Plasma Glucose", 
          xlab="Body Mass Index (kg/m²)", 
          ylab="Plasma Glucose (mg/dL)", 
          col="lightblue", 
          pch=19)

## 5.3 Blood Pressure vs. Diabetes Status
A boxplot was created to visualize the relationship between blood pressure and diabetes status.

     boxplot(`Diastolic Blood Pressure (mm Hg)` ~ `Diabetes Status (1 = Positive, 0 = Negative)`, 
        data = Diabetes_Healthcare_Data, 
        main="Blood Pressure by Diabetes Status", 
        xlab="Diabetes Status (1 = Positive, 0 = Negative)", 
        ylab="Blood Pressure (mm Hg)", 
        col="lightblue")
        
## 5.4 Correlation Between Plasma Glucose and BMI
We computed the correlation coefficient between plasma glucose levels and BMI to identify any linear relationship.

     cor(Diabetes_Healthcare_Data$`Plasma Glucose (mg/dL)`, 
         Diabetes_Healthcare_Data$`Body Mass Index (kg/m²)`)

The correlation coefficient was found to be 0.225, indicating a weak positive correlation.

Data Preprocessing
1. Loading Data
r
Copy code
data <- read.csv("path_to_your_data.csv")
2. Handling Missing Data
r
Copy code
data <- na.omit(data)  # Removing rows with NA values
3. Encoding Categorical Variables
We encoded the target variable Diabetes_Status as a factor.

r
Copy code
data$Diabetes_Status <- factor(data$Diabetes_Status, levels = c(0, 1), labels = c("Negative", "Positive"))
4. Feature Scaling
We scaled the BMI feature to ensure that numerical features are on the same scale for the logistic regression model.

r
Copy code
data$BMI <- scale(data$BMI)
5. Data Splitting
r
Copy code
set.seed(123)
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]
Model Building and Evaluation
1. Model Selection
We used Logistic Regression for predicting diabetes status.

r
Copy code
model <- glm(Diabetes_Status ~ ., data = train_data, family = "binomial")
2. Model Training
We trained the model on the relevant features: Age, BMI, Blood Pressure, Glucose, Pregnancy, and Family History.

r
Copy code
model <- glm(Diabetes_Status ~ Age + BMI + BloodPressure + Glucose + Pregnancy + FamilyHistory, 
             data = train_data, family = "binomial")
3. Model Evaluation
We used performance metrics such as accuracy, precision, recall, F1 score, and ROC curve to evaluate the model.

r
Copy code
predictions <- predict(model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)  # Classification threshold of 0.5
confusionMatrix(predicted_classes, test_data$Diabetes_Status)
Results
1. Model Accuracy and Performance
The logistic regression model performed with an accuracy of 85%. The confusion matrix, accuracy score, and ROC curve are presented below:

r
Copy code
confusion_matrix <- table(Predicted = predicted_classes, Actual = test_data$Diabetes_Status)
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy: ", accuracy))
2. Model Insights
The most significant predictors for diabetes status were Plasma Glucose and BMI.

r
Copy code
summary(model)  # Display model coefficients
3. Visuals
We visualized the model’s performance with an ROC curve:

r
Copy code
library(pROC)
roc_curve <- roc(test_data$Diabetes_Status, predictions)
plot(roc_curve, main = "ROC Curve")
Conclusion
1. Summary of Key Findings
The logistic regression model achieved an accuracy of 85% in predicting diabetes status based on health-related features. The most influential predictors of diabetes were Plasma Glucose and BMI, which were found to have a moderate to strong relationship with the target variable. These results demonstrate the potential of leveraging machine learning for early identification of individuals at risk of diabetes.

2. Implications
This predictive model holds significant promise for healthcare practitioners, enabling early detection of diabetes and facilitating timely interventions for individuals at risk. Integrating this model into healthcare systems could enhance decision-making and preventive care, ultimately reducing the long-term health impact of diabetes. The insights derived from this study underscore the importance of regularly monitoring key health metrics such as plasma glucose and BMI, which are strong indicators of diabetes risk.

3. Interactive Dashboard
For a deeper dive into the dataset and model performance, I’ve created an interactive Tableau dashboard, which visualizes the trends and predictions made in this project. The dashboard allows users to explore the relationships between various features and diabetes status. You can access it [here](https://public.tableau.com/app/profile/yoada.zeleke/viz/UncoveringDiabetesTrendsEDAandPredictiveModeling
