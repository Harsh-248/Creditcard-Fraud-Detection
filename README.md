# Credit Card Fraud Detection

This project is a machine learning model for detecting fraudulent credit card transactions. It uses logistic regression to classify transactions as either legitimate or fraudulent. The dataset for training the model is sourced from Kaggle.

## Features

- Uses **Logistic Regression** for fraud classification.
- **Balanced dataset** by undersampling legitimate transactions.
- **Streamlit UI** for user-friendly fraud detection.
- **Real-time predictions** based on user input.

## Dataset

The dataset can be downloaded from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Steps to Download:
1. Go to the [Kaggle dataset link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Download the `creditcard.csv` file.
3. Place the file in the root directory of this project.

## Installation & Usage

### 1. Clone the Repository

git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

### 2. Run the Application
streamlit run test.py

## How to Use
Run the application using the command above.
Enter transaction features (comma-separated values) in the input field.
Click the Submit button.
The model will predict whether the transaction is Legit or Fraudulent.

## Dependencies
Ensure you have Python installed along with the following libraries:

numpy,
pandas,
scikit-learn,
streamlit

pip install numpy pandas scikit-learn streamlit

Model Information
The dataset is balanced by undersampling non-fraud transactions.
The Logistic Regression model is trained with 80% training data and 20% test data.
Accuracy scores:
Training Accuracy: Displayed in logs
Test Accuracy: Displayed in logs
Contribution
Feel free to contribute! Fork the repo and submit pull requests.
