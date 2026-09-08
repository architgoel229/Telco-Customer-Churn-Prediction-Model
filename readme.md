


# Telco Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the Telco Customer Churn dataset.

The project covers exploratory data analysis, feature investigation, preprocessing, class-imbalance handling, model comparison, model selection, pipeline creation, and a Streamlit application for making predictions on individual customers.

## 🚀 Live Demo

👉 **[Try the Streamlit App](https://telco-customer-churn-prediction-model.streamlit.app/)**

The application allows users to enter customer information and get:
- Churn probability
- Churn / No Churn prediction
- Adjustable decision threshold

---

## 📌 Project Overview

Customer churn is an important problem for subscription-based businesses because identifying customers who are likely to leave can allow the company to take retention actions.

The objective of this project is to build a machine learning model that predicts whether a customer is likely to churn based on their demographic information, tenure, services, contract, and billing information.

The primary focus is on identifying churners effectively, so **recall for the churn class is an important evaluation metric**.

---

## 🗂️ Project Structure

```text
project1/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── model/
│   └── telco_churn_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_featureEngineering_modelComparision.ipynb
│   └── 03_finalPipeline.ipynb
│
├── app.py
├── edaDocument.md
├── report.html
├── requirements.txt
└── runtime.txt

````

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about:

* Customer demographics
* Customer tenure
* Phone services
* Internet services
* Additional internet-related services
* Contract information
* Billing information
* Payment method
* Monthly charges
* Customer churn

The target variable is:

```text
Churn
```

where:

* `Yes` → Customer churned
* `No` → Customer did not churn

---

# 🔎 Exploratory Data Analysis

The first stage of the project focused on understanding the dataset and identifying relationships between customer characteristics and churn.

Some notable patterns observed during EDA were:

### Tenure

Customers with lower tenure showed substantially higher churn rates.

A large proportion of churn occurs during the earlier months of the customer relationship, while churn becomes considerably less frequent at higher tenure values.

### Contract

Contract type showed a strong association with churn.

* Month-to-month customers had the highest churn rate.
* One-year contract customers had considerably lower churn.
* Two-year contract customers had the lowest churn.

Approximately 88.5% of churned customers in the dataset were from month-to-month contracts.

### Internet Service

Churn rates differed considerably across internet-service categories.

Fiber optic customers showed the highest churn rate, while customers with no internet service showed the lowest.

The relationship was also examined jointly with contract type.

For example:

| Contract       |   DSL | Fiber optic | No Internet |
| -------------- | ----: | ----------: | ----------: |
| Month-to-month | 32.1% |      54.48% |      18.64% |
| One year       |  9.3% |      19.29% |       2.48% |
| Two year       | 1.93% |       7.23% |       0.79% |

These values represent observed churn rates within the corresponding segments.

### Monthly Charges

Higher monthly-charge ranges showed higher observed churn in several regions of the dataset.

The relationship was further investigated with contract type and internet service.

### Additional Services

Customers without services such as:

* Online Security
* Online Backup
* Device Protection
* Tech Support

generally showed higher churn rates than customers who had these services.

### Billing and Payment

Electronic check customers showed the highest observed churn among the payment methods.

Paperless billing was also associated with higher churn in the dataset.

These findings describe **observed associations in the dataset and should not be interpreted as causal relationships**.

---

# 🛠️ Feature Engineering & Investigation

Several aspects of the data were investigated before finalizing the model pipeline.

## Outliers

Outliers were investigated using:

* Z-score based analysis
* IQR based analysis
* Distribution visualizations

No major outlier problem was identified that justified removing observations.

Therefore, no aggressive outlier-removal procedure was included in the final pipeline.

---

## Power Transformation

Power transformation was experimented with to investigate whether changing the distributions of numerical features would improve the model.

Although the distributions changed, there was no meaningful improvement in dataset distribution.

Therefore, the transformation was not retained in the final pipeline.

---

## TotalCharges

`TotalCharges` was investigated because it is strongly related to:

```text
tenure × MonthlyCharges
```

Models were compared with and without the feature.

Adding `TotalCharges` did not produce a meaningful improvement in the evaluation metrics.

Therefore, it was excluded from the final feature set.

---

## Internet-Service Segmentation

A separate experiment was performed where different models were trained for:

* Fiber optic customers
* DSL customers
* Customers without internet service

The segmented approach did not provide a meaningful improvement in model performance.

Therefore, the three-model segmentation approach was discarded and a single global model was retained.

---

# ⚙️ Preprocessing

The final preprocessing pipeline uses a `ColumnTransformer`.

Different groups of categorical variables are processed differently depending on their structure.

### One-Hot Encoding

One-hot encoding is used for:

* InternetService
* gender
* PaymentMethod
* Contract

### Ordinal Encoding + MinMax Scaling

Binary variables such as:

* Partner
* Dependents
* PhoneService
* PaperlessBilling

are ordinal encoded and scaled.

Variables with three possible states such as:

* MultipleLines
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies

are also ordinal encoded using their corresponding category ordering and scaled.

### Numerical Features

`tenure` and `MonthlyCharges` are processed using:

```text
PolynomialFeatures(degree=3)
        ↓
StandardScaler
```

This allows the final model to work with polynomially expanded numerical features.

---

# ⚖️ Handling Class Imbalance

The dataset contains fewer churned customers than non-churned customers.

To address the class imbalance during training, **SMOTE (Synthetic Minority Over-sampling Technique)** was incorporated into the final pipeline.

The final pipeline therefore performs:

```text
Raw customer data
        ↓
Preprocessing
        ↓
SMOTE
        ↓
Stacking Classifier
```

SMOTE is applied only during the training process through the `imblearn` pipeline.

---

# 🤖 Model Comparison

Several classification algorithms were investigated, including:

* Logistic Regression
* Gaussian Naive Bayes
* K-Nearest Neighbors
* Support Vector Machine
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost
* Stacking Classifier

The models were evaluated using metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Because missing an actual churner can be costly from a customer-retention perspective, **recall for the churn class was given particular importance**.

At the same time, precision and F1-score were considered to avoid increasing recall at the cost of an excessive number of false positives.

---

# 🧠 Final Model

The final model is a **Stacking Classifier** consisting of:

* AdaBoost
* Gaussian Naive Bayes
* Random Forest
* XGBoost
* Gradient Boosting

with:

```text
Logistic Regression
```

as the final estimator.

The final model is combined with preprocessing and SMOTE into a single pipeline.

Conceptually:

```text
                 ┌── AdaBoost
                 ├── Naive Bayes
Input → Preprocess → SMOTE → Stacking ├── Random Forest
                 ├── XGBoost
                 └── Gradient Boosting
                          ↓
                   Logistic Regression
```

---

# 📈 Final Model Performance

On the held-out test set, the final pipeline produced:

```text
Accuracy: 0.77
```

For the churn class (`1`):

```text
Precision: 0.59
Recall:    0.64
F1-score:  0.62
```

Confusion matrix:

```text
[[831 176]
 [143 257]]
```

The test set contained:

```text
1007 non-churn customers
400 churn customers
```

---

# 🎯 Threshold Analysis

The model outputs churn probabilities using:

```python
predict_proba()
```

The ROC curve was also used to investigate a decision threshold.

A threshold was selected using:

```python
np.argmax(tpr - fpr)
```

This resulted in a threshold that increased recall for the churn class.

With the adjusted threshold:

```text
Accuracy:  0.75
Recall:     0.79
Precision:  0.54
F1-score:   0.64
```

Confusion matrix:

```text
[[741 266]
 [ 85 315]]
```

This demonstrates the trade-off involved in lowering the classification threshold:

```text
Lower threshold
       ↓
More customers predicted as churners
       ↓
Higher recall
       ↓
More false positives
```

The Streamlit application is adjusted with the ROC curve driven threshold.

---

# 🚀 Streamlit Application

The project includes a Streamlit application that allows a user to enter information about an individual customer and obtain a churn prediction.

The application:

1. Accepts customer information.
2. Creates a DataFrame containing the original input features.
3. Passes the data through the exported machine learning pipeline.
4. Obtains the churn probability.
5. Applies a selected decision threshold.
6. Displays the predicted result.

The model itself handles preprocessing and SMOTE internally through the exported pipeline.

Run the application using:

```bash
streamlit run app.py
```

---

# 💾 Model Deployment

The complete trained pipeline is serialized using `joblib`.

```python
joblib.dump(pipeline, "model/telco_churn_model.pkl")
```

The exported object contains:

```text
Preprocessing
      ↓
SMOTE
      ↓
Trained Stacking Classifier
```

Therefore, new customer data can be passed directly to the exported pipeline without manually reproducing the preprocessing steps.

The model can be loaded using:

```python
import joblib

model = joblib.load("model/telco_churn_model.pkl")
```

and predictions can be obtained using:

```python
prediction = model.predict(input_data)
```

or:

```python
probability = model.predict_proba(input_data)
```

The exported model was reloaded and compared against the original pipeline to verify that both produced identical predictions.

---

# 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Joblib
* Streamlit
* Jupyter Notebook

---

# 📁 Notebooks

### `01_EDA.ipynb`

Contains the exploratory data analysis and investigation of relationships between customer characteristics and churn.

### `02_featureEngineering_modelComparision.ipynb`

Contains:

* Feature investigations
* Transformation experiments
* Model experiments
* Hyperparameter investigations
* Threshold analysis
* TotalCharges investigation
* Internet-service segmentation experiment
* Model comparison

### `03_finalPipeline.ipynb`

Contains the finalized:

* Preprocessing
* SMOTE
* Stacking Classifier
* Model training
* Evaluation
* Model serialization

---

# ▶️ Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/architgoel229/Telco-Customer-Churn-Prediction-Model
```

## 2. Create a virtual environment

```bash
python -m venv myenv
```

Activate it on Linux:

```bash
source myenv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt 
```

## 4. Run the application

```bash
streamlit run app.py
```

---

# 📌 Key Takeaways

The project showed that:

* Tenure is strongly associated with churn.
* Month-to-month contracts have substantially higher observed churn.
* Internet service and contract type show important differences in churn rates.
* Several additional services are associated with lower observed churn.
* `TotalCharges` did not provide enough additional predictive value to retain it.
* Segmenting the model by internet service did not meaningfully improve performance.
* SMOTE was used to address class imbalance during model training.
* A stacking ensemble was selected as the final model.
* Classification threshold affects the trade-off between recall and false positives.
* The entire preprocessing and model workflow was packaged into a single deployable pipeline.

---

# ⚠️ Important Note

The relationships discussed in the analysis are **associations observed in the dataset**.

They should not be interpreted as evidence that a particular customer characteristic directly causes churn.

The model is intended as a predictive tool for identifying customers who may be at higher risk of churn, not as a causal explanation of why customers leave.

---

## 👤 Author

**Archit Goel**

B.Tech Student | Machine Learning & Python

---

## 📜 License

This project is intended for educational and portfolio purposes.

