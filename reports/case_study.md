# Customer Churn Prediction Case Study

## Executive Summary

This project builds a supervised machine learning solution to identify telecom customers likely to churn. The final Random Forest model achieved a ROC-AUC of 0.8430 and recall of 0.7834, making it useful for prioritizing customer retention outreach.

## Business Context

Telecom companies lose revenue when customers cancel service. Retention teams need a way to identify at-risk customers before cancellation so they can target outreach, incentives, and support resources more efficiently.

## Approach

The workflow cleaned the IBM Telco Customer Churn dataset, transformed categorical service and billing fields with one-hot encoding, scaled numeric fields, and compared Logistic Regression, Decision Tree, and Random Forest classifiers.

Accuracy was not treated as the main success metric. Churn is a business intervention problem, so the evaluation emphasized ROC-AUC, recall, precision, F1, and the confusion matrix.

## Model Selection

Random Forest was selected because it delivered the strongest ROC-AUC while maintaining high recall.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7544 | 0.5251 | 0.7834 | 0.6288 | 0.8430 |
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8416 |
| Decision Tree | 0.7544 | 0.5260 | 0.7567 | 0.6206 | 0.8318 |

## Business Findings

The strongest churn patterns were concentrated around contract flexibility, billing behavior, service type, and customer tenure.

- Month-to-month customers had a churn rate of 42.7%.
- Electronic check customers had a churn rate of 45.3%.
- Fiber optic customers had a churn rate of 41.9%.
- The dataset showed $139,130.85 in observed monthly charges associated with churned customers.

## Retention Strategy

Recommended actions:

- Offer annual contract incentives to high-risk month-to-month customers.
- Create onboarding check-ins for customers in the first 12 months.
- Promote automatic payment options to customers using electronic checks.
- Bundle technical support for high-risk customers without support.
- Use churn probability to prioritize retention queues.

## Recruiter Signal

This project demonstrates predictive modeling, classification evaluation, model interpretation, business translation, and dashboard deployment. It is designed to show the ability to move from historical customer data to forward-looking business decisions.

