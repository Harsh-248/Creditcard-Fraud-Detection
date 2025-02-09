import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Split the data into legit and fraud classes
legit = data[data.Class == 0]
fraud = data[data['Class'] == 1]

# Create a balanced dataset by sampling the legit class to match the fraud class
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split features and target
X = data.drop("Class", axis=1)
Y = data["Class"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Calculate accuracy scores
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

# Streamlit UI setup
st.title("Credit Card Fraud Detection Model")

# Input feature prompt for the user
input_df = st.text_input('Input all features (comma-separated values)')

# Process input when the user submits
submit = st.button("Submit")

if submit:
    # Convert the input string to a list of floats
    input_df_lst = input_df.split(',')
    
    # Convert the list of strings to a numpy array of floats
    features = np.asarray(input_df_lst, dtype=np.float64)
    
    # Ensure the features array is reshaped to match the model's expected input shape
    prediction = model.predict(features.reshape(1, -1))
    
    if prediction[0] == 0:
        st.write("Legit Transaction")
    else:
        st.write("Fraud Transaction")
