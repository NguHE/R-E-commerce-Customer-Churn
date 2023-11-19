# E-commerce-Customer-Churn
E-commerce customer churn in R programming

Applied Machine Learning course

## Data Source
Link: [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

## Introduction
- E-comm: Subset of electronic business, which encompasses the use of digital processes to conduct business online 24/7
- Advantageous: Expanded market reach, efficient distribution systems, reduced costs, effective advertising and marketing, enhanced customer flexibility, easy comparison with competitors, swift custoemr service
- Predict risk of customer churn enables customer to proactively address the problems and mitigate churn rates

## Problem Statement
- United States corporation lose a starggering USD 136.8 billion annually due to customer attrition
- Higher cost (five to six-fold in expenses) imposed on getting new customers than keeping current one 
- Factors: customer satisfaction, service quality, experience, trust and brand fidelity
- Effect: Impact revenue, proditability
- Importance: Foster long-lasting customer relationships and loyalty

## Aim and Objectives
The aim of this project is to develop a predictive model using ML algorithms for analysing customer attrition in the E-comm industry. 

1.	To identify the key features that affect E-comm customer attrition.
2.	To identify the optimal algorithm from LR, DT and RF that can accurately predict customer churn based on evaluation of performance metrics.
3.	To provide customer retention strategies based on insights gained from the analysis.

## Methodology
- CRISP-DM methodology
<img src="https://github.com/NguHE/E-commerce-Customer-Churn/assets/125574265/0f401ed2-610d-4aaa-868a-9a331b799bb7" width="300" />

![image](https://github.com/NguHE/E-commerce-Customer-Churn/assets/125574265/3ab5cbb0-fc6e-43ed-b008-35346d61e9bb)

1. Type I Error (False Positive): Spending money on customers who are not the right fit.
2. Type II Error (False Negative): Losing loyal customer.

## Results
| Dataset               | Algorithm | Test/Training | Accuracy | Precision | Recall | F1-Score | AUC   |
|-----------------------|-----------|---------------|----------|-----------|--------|----------|-------|
| Original dataset      | LR        | Training      | 89.39%   | 76.82%    | 52.90% | 62.66%   | -     |
|                       |           | Test          | 89.43%   | 77.52%    | 52.63% | 62.70%   | 90.26%|
|                       | DT        | Training      | 88.90%   | 74.16%    | 52.24% | 61.30%   | -     |
|                       |           | Test          | 88.99%   | 72.60%    | 55.79% | 63.10%   | 80.95%|
|                       | RF        | Training      | 100.00%  | 100.00%   | 100.00%| 100.00%  | -     |
|                       |           | Test          | 97.51%   | 97.65%    | 87.37% | 92.22%   | 93.47%|
| Oversampling dataset  | LR        | Training      | 81.26%   | 80.69%    | 82.19% | 81.43%   | -     |
|                       |           | Test          | 81.26%   | 46.96%    | 85.26% | 60.56%   | 90.19%|
|                       | DT        | Training      | 82.33%   | 80.99%    | 84.49% | 82.70%   | -     |
|                       |           | Test          | 83.30%   | 50.31%    | 85.26% | 63.28%   | 86.47%|
|                       | RF        | Training      | 100.00%  | 100.00%   | 100.00%| 100.00%  | -     |
|                       |           | Test          | 98.49%   | 95.77%    | 95.26% | 95.51%   | 97.20%|
| Hyperparameter on oversampled data set | LR        | Test          | 81.26%   | 46.96%    | 85.26% | 60.56%   | 90.12%|
|                       | DT        | Test          | 90.85%   | 94.21%    | 66.05% | 77.66%   | 94.03%|
|                       | RF        | Test          | 96.71%   | 93.16%    | 88.06% | 90.54%   | 99.52%|


**Performance metrics comparison with other studies using the same dataset**
| Reference               | Model                           | Accuracy | Precision | Recall | F1    | AUC   |
|-------------------------|---------------------------------|----------|-----------|--------|-------|-------|
| In this study            | RF with hyperparameter tuning    | 96.71%   | 93.16%    | 88.06% | 90.54%| 99.52%|
| Alshamsi (2022)          | RF                              | 93.50%   | -         | -      | -     | -     |
|                         | LR                              | 80.50%   | -         | -      | -     | -     |
|                         | DT                              | 84.80%   | -         | -      | -     | -     |
| Awasthi (2022)           | Stacking                        | 98.00%   | 91.00%    | 92.00% | 93.00%| -     |
|                         | DT                              | 97.02%   | 90.52%    | 95.72% | 92.98%| 97.91%|
|                         | KNN                             | 97.04%   | 92.30%    | 95.45% | 92.85%| 96.99%|
|                         | SVM                             | 89.87%   | 76.27%    | 51.15% | 61.22%| 74.08%|
|                         | RF                              | 90.58%   | 98.91%    | 39.72% | 56.91%| 69.88%|
| Kiran et al. (2022)      | LDA                             | 92.53%   | -         | -      | -     | -     |
|                         | NB                              | 90.31%   | -         | -      | -     | -     |
|                         | SVM                             | 86.94%   | -         | -      | -     | -     |

![image](https://github.com/NguHE/E-commerce-Customer-Churn/assets/125574265/8d086c22-3e31-460d-bec6-3548e9f02ce7)

![image](https://github.com/NguHE/E-commerce-Customer-Churn/assets/125574265/dd2e9fc7-c726-4deb-8bdd-b11add3179b2)



## Recommendations
1. Loyalty Program with Cashback Rewarding System
2. Survey and Targeted Promotions
3. Customer Service
4. Customer Satisfaction
5. Product Delivery Efficiency
6. Personalized Marketing
7. Dashboard to monitor and visualize

## Conclusion
- SMOTE and hyperparameter tuning improve model performance
- RF demonstrated the best performance

## Future Work
- Work with larger samples
- Perform customer segmentation based on shopping behaviours
- Include additional variables such as user experience and product quality to further enhance predictive power of model
- Explore other algorithm such as boosting or bagging

## References for Comparison
Alshamsi, A. (2022). Customer Churn prediction in ECommerce Sector. Thesis. Rochester 
Institute of Technology. Retrieved June 20, 2023, from https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=12319&context=theses

Awasthi, S. (2022). Customer Churn Prediction on E-Commerce Data using Stacking 
Classifier. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.20291694.v1 

Kiran, C.N., Vailshery, W.S., Patil, S.A. (2022). Research on customer prediction in mobile 
commerce using supervised model. Journal of Tianjin University Science and Technology, 55(5), 29-43.
