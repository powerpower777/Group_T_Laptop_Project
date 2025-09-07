import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import re

# Defining dataset to a variable
df = pd.read_csv("cleaned_laptop_price_dataset.csv")

# Let's check what's actually in the Ram column
print("Unique values in Ram column:", df['Ram'].unique())

# Clean the Ram column - extract only the numeric part before any space or text
def clean_ram(value):
    if isinstance(value, str):
        # Extract the numeric part at the beginning of the string
        match = re.match(r'^(\d+)', value)
        if match:
            return int(match.group(1))
    return value

# Apply cleaning to the Ram column
df['Ram'] = df['Ram'].apply(clean_ram)

# Creation And Training Models
x = df.drop(columns =["Price", "Inches", "Weight"], axis=1 ) 
y = df["Price"] # target variable

x_train, x_test, y_train, y_test = split(x, y, test_size= 0.35, random_state=0)

# Identify categorical & numeric columns
categorical_columns = x.select_dtypes(include=['object']).columns
numerical_columns = x.select_dtypes(exclude=['object']).columns

# Transformer: Encode categorical + scale numeric
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
    # Clean the Ram value in user input
    cleaned_input = user_input.copy()
    ram_value = cleaned_input[5]  # Ram is the 6th element (index 5)
    if isinstance(ram_value, str):
        match = re.match(r'^(\d+)', ram_value)
        if match:
            cleaned_input[5] = int(match.group(1))
    
    # Create a dictionary with the correct column names and user input
    new_data = {col: [value] for col, value in zip(x.columns, cleaned_input)}
    
    # Create DataFrame with the same structure as training data
    new_df = pd.DataFrame(new_data)
    
    # Use the model to predict
    return model.predict(new_df)

st.title("Group T Laptop Price Project")
st.write("Choose the laptop features to know the price")

# Creation of selection box to get user input
Company = st.selectbox("Company", sorted(list(set(x["Company"].tolist()))))
Product = st.selectbox("Product", sorted(list(set(x["Product"].tolist()))))
TypeName = st.selectbox("Type", sorted(list(set(x["TypeName"].tolist()))))
ScreenResolution = st.selectbox("Screen Resolution", sorted(list(set(x["ScreenResolution"].tolist()))))
Cpu = st.selectbox("CPU", sorted(list(set(x["Cpu"].tolist()))))

# For Ram, we need to extract just the numeric values for display
ram_options = sorted(list(set(x["Ram"].astype(str).tolist())))
Ram = st.selectbox("RAM", ram_options)

Memory = st.selectbox("Memory", sorted(list(set(x["Memory"].tolist()))))
Gpu = st.selectbox("GPU", sorted(list(set(x["Gpu"].tolist()))))
Operating_System = st.selectbox("Operating_System", sorted(list(set(x["Operating_System"].tolist()))))

user_input = [Company, Product, TypeName, ScreenResolution, Cpu, Ram, Memory, Gpu, Operating_System]

# shows the price of the device
if st.button("Click"):
    try:
        price = get_price(user_input)[0]
        text = f"The price is £{price:,.2f}"
        st.write(text)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
