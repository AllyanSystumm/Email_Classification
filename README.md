# Email Spam Classifier

This project is a machine learning-based **Email Spam Classifier** that classifies emails as **Spam** or **Not Spam** based on the email content. The model is trained using two machine learning algorithms: **Naive Bayes** and **Logistic Regression**.

The application is deployed as a **Streamlit** app, allowing users to enter email content and get predictions.

## Table of Contents
1. [Data Collection](#data-collection)
2. [Text Preprocessing](#text-preprocessing)
3. [Feature Extraction](#feature-extraction)
4. [Model Selection and Training](#model-selection-and-training)
5. [Model Evaluation](#model-evaluation)
6. [Deployment with Streamlit](#deployment-with-streamlit)
7. [How to Run the Application](#how-to-run-the-application)
8. [Dependencies](#dependencies)
9. [Contributors](#contributors)

## Data Collection
The dataset used for training the model was collected from **Kaggle**. Specifically, the **SMS Spam Collection** dataset, which contains a list of SMS messages classified as either **spam** or **ham** (not spam). The dataset was downloaded in **CSV format** and contains the following columns:
- `Category`: This column contains the labels, either `spam` or `ham`.
- `Message`: This column contains the text of the SMS message.

You can access the dataset on Kaggle via the following link:
[SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

### Data Collection Code
To load the data, we use the `pandas` library in Python:
<img width="1500" height="768" alt="email" src="https://github.com/user-attachments/assets/bb8099ee-4b39-496b-8f27-734a9d2d7f6a" />


```python
import pandas as pd

# Load the dataset
   data = pd.read_csv('data/mail_data.csv') 

Run command : streamlit run src/app.py

<img width="1500" height="768" alt="email" src="https://github.com/user-attachments/assets/3e6de098-5c02-4dcc-9632-9d4220db0a0f" />


