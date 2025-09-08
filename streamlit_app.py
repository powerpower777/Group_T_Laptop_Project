import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import os

# Defining dataset to a variable
try:
    df = pd.read_csv("cleaned_dataset.csv")
except FileNotFoundError:
    st.error("Error: 'cleaned_dataset.csv' not found. Please ensure the file is in the same directory as the app.")
    st.stop()

# Drop any unnamed columns and strip whitespace from column names
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

# Creation And Training Models
x = df.drop(columns =["Price", "Inches", "Weight"], axis=1) 
y = df["Price"] # target variable

x_train, x_test, y_train, y_test = split(x, y, test_size=0.15, random_state=8)

# Identify categorical & numeric columns
categorical_columns = x.select_dtypes(include=['object']).columns
numerical_columns = x.select_dtypes(exclude=['object']).columns

#Transformer: Encode categorical + scale numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('category', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('numbers', StandardScaler(), numerical_columns)
    ]
)

# Pipeline: preprocessing + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model.fit(x_train, y_train)

# Function to predict laptop price
def get_price(user_input):
    """
    Predicts the laptop price based on user input.
    The function creates a DataFrame from the input and passes it directly
    to the trained pipeline for prediction.
    """
    # Create a DataFrame from the user's input list, using the original
    # training columns for the headers.
    user_data_df = pd.DataFrame([user_input], columns=x.columns)
  
    # The pipeline handles both the preprocessing (encoding and scaling)
    # and the prediction in one step.
    predicted_price = model.predict(user_data_df)
  
    return predicted_price


st.title("Group T Laptop Price Project")
st.write("Choose the laptop features to know the price")

# Creation of selection box to get user input

Company = st.selectbox("Company", sorted(list(set(x["Company"].tolist()))))
Product = st.selectbox("Product", sorted(list(set(x["Product"].tolist()))))
TypeName = st.selectbox("Type", sorted(list(set(x["TypeName"].tolist()))))
ScreenResolution = st.selectbox("Screen Resolution", sorted(list(set(x["ScreenResolution"].tolist()))))
Cpu = st.selectbox("CPU", sorted(list(set(x["Cpu"].tolist()))))
Ram = st.selectbox("RAM", sorted(list(set(x["Ram"].tolist()))))
Memory = st.selectbox("Memory", sorted(list(set(x["Memory"].tolist()))))
Gpu = st.selectbox("GPU", sorted(list(set(x["Gpu"].tolist()))))
Operating_System = st.selectbox("Operating_System", sorted(list(set(x["Operating_System"].tolist()))))

user_input = [Company, Product, TypeName, ScreenResolution, Cpu, Ram, Memory, Gpu, Operating_System]

# shows the price of the device
if st.button("Calculate Price"):
    predicted_price = get_price(user_input)[0]
    # Check for negative price predictions and handle them
    if predicted_price < 0:
        st.write("Sorry, the model predicted a negative price. This can sometimes happen with this type of model due to the complexity of the data.")
    else:
        text = f"The estimated price is £{predicted_price:,.2f}"
        st.write(text)
