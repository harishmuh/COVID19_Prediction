<p align="center">
  <img src="https://kominfosandi.kamparkab.go.id/wp-content/uploads/2021/08/banner.png" alt="COVID19" width="600">
</p>


# **Machine Learning for COVID-19 Positivity Prediction**

### **Project Background & Context**
The COVID-19 pandemic has underscored the urgent need for efficient methods to identify potentially infected individuals. Although PCR and antigen tests remain the gold standards for diagnosis, these procedures are often resource-intensive, time-consuming, and not always accessible—particularly in remote or resource-limited regions.

The dataset (about 280 thousand individuals) provided by the Ministry of Health (source: [data.go.il](https://data.gov.il/dataset/covid-19)) contains key patient information, including:

* Demographics (e.g., age group, gender)

* Symptoms (e.g., cough, fever, sore throat, shortness of breath, headache)

* Exposure history or testing indication (e.g., contact with confirmed cases, recent travel, or other test indications)

* Laboratory-confirmed test results

### **Problem Statement**

* How do demographic, symptom, and exposure features influence COVID-19 test results?

* Which machine learning model achieves the best performance on unseen data, and what are the best model parameters?

* Which features contribute most to prediction accuracy?


### **Analytical Approach**

This analysis aims to uncover patterns that differentiate patients who test positive for COVID-19 from those who test negative. The model should be able to predict based on the given historical data from the hospital/healthcare institutions. Besides, we want to know which variables are crucial in increasing the probability of getting a positive test result. There are several steps in the approach to achieve our objective that consist of:

* Data Understanding & Preparation – Explore the dataset, check missing values, and perform encoding for categorical variables.

* Feature Engineering – Transform features (e.g., impute missing demographics, create binary symptom indicators).

* Model Development – Train multiple classification models (e.g., logistic regression, random forest, gradient boosting, etc.) to predict COVID-19 positivity.

* Model Evaluation – Compare and evaluate model performances using metrics such as accuracy, precision, recall, F1-score, ROC-AUC curve, and precision-recall curve.

* Hyperparameter tunning - Perform further improvement of the models. Select the top 3 best models and conduct hyperparameter tunning using randomized search to find the optimum parameters.

* Interpretability – Identify the most influential features driving predictions (e.g., feature importance, permutation importance for global explanation, and LIME for local explanation).

**Metric evaluation**

In this study, the target variable represents the COVID-19 test result:

Target:

* 0 : Patient tested negative for COVID-19
* 1 : Patient tested positive for COVID-19

Error Types:

* Type I Error (False Positive) – The model predicts a patient as COVID-19 positive when they are not.
  * Impact: Leads to unnecessary testing and resource usage but poses minimal health risk.

* Type II Error (False Negative) – The model predicts a patient as negative when they are actually infected.
  * Impact: Poses serious health and safety risks since undetected infected individuals may deteriorate and/or spread the virus further.
 
Given these implications, Recall is prioritized as the main performance metric because it measures how effectively the model captures all actual positive cases.

The recall formula is:


> **Recall** = True Positive / True Positive + False Negative


A higher recall value indicates the model is better at identifying COVID-19-positive patients, reducing the risk of missed infections. To complement recall, the Precision-Recall (PR) Curve is also analyzed to evaluate how well the model balances the trade-off between capturing true positives and avoiding false alarms, especially under class imbalance conditions.


### **Data Understanding**

#### **Exploratory Data Analysis**

**Demographic**

<img width="1688" height="539" alt="image" src="https://github.com/user-attachments/assets/48a90486-89d5-46f8-98c4-d325aafdc103" />

<img width="1696" height="509" alt="image" src="https://github.com/user-attachments/assets/9b9e1858-d098-4b33-9cef-2d563f0d0971" />


**Symptoms**

<img width="1694" height="535" alt="image" src="https://github.com/user-attachments/assets/eec7cdf8-5904-4569-aadf-4a928ed60fe2" />

<img width="1683" height="521" alt="image" src="https://github.com/user-attachments/assets/7b28536e-2f96-4238-ab73-50228fe3056e" />

<img width="1711" height="538" alt="image" src="https://github.com/user-attachments/assets/db78c8b7-c62b-43bb-a440-dc429b325f74" />

<img width="1699" height="523" alt="image" src="https://github.com/user-attachments/assets/ae16d8b5-066f-4be4-b7cf-4f4217822b5c" />

<img width="1680" height="516" alt="image" src="https://github.com/user-attachments/assets/d9a54e7b-c6b5-457b-b534-1e370f32fe02" />

**Exposure history or testing indication**

<img width="1690" height="505" alt="image" src="https://github.com/user-attachments/assets/9ae106c2-959e-43b3-8816-d501ad529cb6" />

**Target distribution**

<img width="613" height="646" alt="image" src="https://github.com/user-attachments/assets/05f553ff-038e-40a3-bb25-6b2973bcb735" />



### **Modelling**

We trained and tested 8 models with their basic parameters.

<img width="839" height="417" alt="image" src="https://github.com/user-attachments/assets/e9734fce-7548-4a8b-bfd0-bc367e6dfc80" />

Based on the result above, we can provide general observations:
* Ensemble tree-based models (XGBoost, LightGBM, Random Forest) dominate recall performance.
* Boosting methods (XGBoost, LightGBM) slightly outperform bagging (Random Forest), showing they capture minority class patterns better.
* Models like AdaBoost and Logistic Regression show lower recall, suggesting those models are relatively underfitting or poor handling of class imbalance.

**Hyperparameter tuning**

We selected the top 3 models (XGBoost, LGBM, and Random Forest) and conducted hyperparameter tuning to further improve the model performance.

**Recall comparison**

| Model          | Training Set | Testing Set |
|----------------|-----------------------------|-----------------------------|
| **XGBoost (Before Tuning)**   | 0.799 | 0.789 |
| **XGBoost (After Tuning)**    | 0.821 | 0.813 |
| **LightGBM (Before Tuning)**  | 0.797 | 0.789 |
| **LightGBM (After Tuning)**   | 0.818 | 0.804 |
| **Random Forest (Before Tuning)** | 0.795 | 0.786 |
| **Random Forest (After Tuning)**  | 0.800 | 0.789 |


**Overall Model Performance After Tuning**

| Model         | Recall (Test) | Precision (Class 1) | ROC-AUC | PR-AUC | Key Insight |
|----------------|---------------|----------------------|----------|---------|--------------|
| **XGBoost**    | 0.813 | 0.29 | 0.896 | 0.629 | Highest recall; good at detecting positives but many false positives (low precision). |
| **LightGBM**   | 0.804 | 0.32 | 0.901 | 0.660 | Better balance between recall and precision; good generalization. |
| **Random Forest** | 0.789 | 0.36 | 0.902 | 0.671 | Most robust model overall. Best AUC and precision–recall balance. |

While XGBoost maximizes recall (catching most positive cases), it sacrifices precision, resulting in more false positives. The LGBM is slightly better than XGBoost with a better balance of recall and precision. In contrast, Random Forest offers the most stable and balanced performance across all metrics, making it the most reliable model for prediction.

**Recommended final model:** Random Forest, as this model demonstrated the best trade-off between recall, precision, and AUC metrics in comparison to other models.

### **Model evaluation**



<img width="1082" height="488" alt="image" src="https://github.com/user-attachments/assets/96abaa56-9965-4250-889c-10e1afb3e7a3" />

The learning curve above shows that our selected best model, the tuned Random Forest, achieves stable performance as training data increases. Both training and validation recall scores converge around 0.80–0.81, indicating good generalization and minimal overfitting.

<img width="1477" height="718" alt="image" src="https://github.com/user-attachments/assets/9897ca68-e162-4ac3-bc24-9530ba25aa00" />



<img width="915" height="685" alt="image" src="https://github.com/user-attachments/assets/b7c89b4c-fcca-4ce7-94cf-eef8ae3818c3" />

Permutation importance provides a global explanation of a model by measuring feature contribution to the overall model. The permutation importance plot above shows that `test_indication` is the most influential feature, followed by `age_60_and_above` and `fever`,


**Local Interpretable Model-agnostic Explanations (LIME) - Local explanation**

<img width="1611" height="588" alt="image" src="https://github.com/user-attachments/assets/267d3c52-ccf0-4caa-83cb-68350d0e75ba" />

<img width="1281" height="638" alt="image" src="https://github.com/user-attachments/assets/76c2ec4a-ec6d-4927-aa80-664c17e53a3a" />

LIME model predicted the patient (index 230676) as COVID-19 positive (79% probability). Although the patient does not show several common symptoms (no cough, sore throat, shortness of breath, or headache), the model strongly focuses on exposure risk/history. Specifically, the feature “test_indication = Contact with confirmed case” and the presence of fever are the most influential factors pushing the prediction toward positive. These two features outweighed the negative effects from the absence of other symptoms. 

### **Conclusion**

**1. Influence of Demographic, Symptom, and Exposure History Features**

* Exposure history is the strongest predictor of COVID-19 results.
    * The feature `test_indication`: Contact with a confirmed case has the highest importance, showing that contact with a confirmed case is the key driver of positive outcomes.

    * `Age` also plays a significant role, as people aged 60 and above have higher chances of testing positive.

    * Symptoms such as `fever`, `cough`, `sore throat`, `shortness of breath`, and `headache` also contribute to the model but with less impact. less to the model.
This suggests that symptom presence alone is not enough to predict infection accurately without exposure and age information.

**2. Best-Performing Model**

* The tuned Random Forest classifier achieved the best overall performance on unseen data with parameters as
    * n_estimators: 100 
    * min_samples_split: 2 
    * min_samples_leaf: 1
    * max_features: 'sqrt'
    * max_depth: 10 
    * model__bootstrap: True

* The learning curve of the model also shows that both training and validation recall scores improve as sample size increases, with only a small gap between them. This indicates the model is well-generalized (not overfitting) and performs reliably in predicting COVID-19 outcomes.

**3. Key Features for Prediction Accuracy**

Based on Feature Importance, Permutation Importance (global explanation), and LIME (local explanation) analyses confirm that:

* `Test indication` (especially “Contact with confirmed”) is the most critical variable.

* `Age 60 and above` and `Fever` follow as secondary contributors.

* Other features like `gender`, `headache`, `sore throat`, and `shortness of breath` have minimal influence on recall performance.


### **Asset**
* [Notebook (ipynb)]() 











