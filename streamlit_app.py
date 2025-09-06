import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import streamlit as st

# Hardcoding a sample of the dataset to make the script self-contained.
data = {
    'Company': ['Apple', 'Dell', 'HP', 'Lenovo', 'Apple', 'Dell'],
    'Product': ['Macbook Pro', 'XPS 13', 'Spectre x360', 'Yoga C930', 'Macbook Air', 'Inspiron'],
    'TypeName': ['Ultrabook', 'Ultrabook', '2 in 1 Convertible', '2 in 1 Convertible', 'Ultrabook', 'Notebook'],
    'Inches': [13.3, 13.3, 13.3, 13.9, 13.3, 15.6],
    'ScreenResolution': ['Retina Display', '4K Ultra HD', 'Full HD', '4K Ultra HD', 'Retina Display', 'Full HD'],
    'Cpu': ['Intel Core i7', 'Intel Core i7', 'Intel Core i7', 'Intel Core i7', 'Intel Core i5', 'Intel Core i5'],
    'Ram': ['16GB', '8GB', '16GB', '16GB', '8GB', '8GB'],
    'Memory': ['256GB SSD', '512GB SSD', '1TB SSD', '1TB SSD', '128GB SSD', '1TB HDD'],
    'Gpu': ['Intel Iris Plus Graphics', 'Intel Iris Plus Graphics', 'Nvidia GeForce MX150', 'Intel UHD Graphics 620', 'Intel Iris Plus Graphics', 'AMD Radeon 520'],
    'Operating_System': ['macOS', 'Windows 10', 'Windows 10', 'Windows 10', 'macOS', 'Windows 10'],
    'Weight': [1.37, 1.29, 1.32, 1.35, 1.25, 2.3],
    'Price': [1339.69, 1495.59, 1749.69, 1999.99, 1158.69, 449.69]
}

df = pd.DataFrame(data)

# --- Model Creation and Training ---
# The rest of the model training pipeline is correct and self-contained
x = df.drop(columns=["Price", "Inches", "Weight"], axis=1) 
y = df["Price"] # target variable

x_train, x_test, y_train, y_test = split(x, y, test_size=0.35, random_state=0)

categorical_columns = x.select_dtypes(include=['object']).columns
numerical_columns = x.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('category', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('numbers', StandardScaler(), numerical_columns)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(x_train, y_train)

def get_price(user_input_list):
    """
    Predicts the price of a laptop based on user input using the trained pipeline.
    """
    user_data = pd.DataFrame([user_input_list], columns=x.columns)
    predicted_price = model.predict(user_data)
    return predicted_price[0]


# --- Streamlit Web App Interface ---
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

# Create the user input list from the Streamlit selections
user_input = [Company, Product, TypeName, ScreenResolution, Cpu, Ram, Memory, Gpu, Operating_System]

# shows the price of the device when the button is clicked
if st.button("Click"):
    try:
        predicted_price = get_price(user_input)
        text = f"The price is £{predicted_price:,.2f}"
        st.write(text)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure your input values exactly match the examples provided.")
