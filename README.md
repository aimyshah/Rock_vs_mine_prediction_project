**Project Overview**

This project uses machine learning to predict whether an object detected by sonar signals is a rock or a mine.

**Problem Statement**

Sonar signals are commonly used in underwater object detection. Based on the reflected sonar signals, the goal is to classify the object as:
Rock (R)
Mine (M)
Correct classification is important for applications such as naval safety and underwater exploration.

**Dataset**

-The dataset consists of numerical features extracted from sonar signals.  
-Each row represents a sonar signal bounced off an object.
-The target variable contains:  
R → Rock  
M → Mine  

**Approach**

-Loaded and explored the dataset.   
-Converted categorical labels into numerical values.  
-Split the data into training and testing sets.   
Trained a Logistic Regression model.  
Evaluated the model using accuracy score.  
***Why Logistic Regression?***
The problem is a binary classification task.  
Logistic Regression is simple, efficient, and performs well on linearly separable data. 

**Results**

Training Accuracy: 89.5%  
Testing Accuracy: 85.3%  
