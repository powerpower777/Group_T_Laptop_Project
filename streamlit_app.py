import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st

# Defining dataset to a variable by reading the uploaded CSV file.
df = pd.read_csv("cleaned_laptop_price_dataset.csv")

# Creation And Training Models
x = df.drop(columns =["Price", "Inches", "Weight"], axis=1 ) 
y = df["Price"] # target variable

x_train, x_test, y_train, y_test = split(x, y, test_size= 0.15, random_state=8)

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

# Corrected Function to predict laptop price
def get_price(user_input, fallback_price):
   """
   Predicts the price of a laptop. If the model fails due to a ValueError,
   it returns a fallback price to prevent the app from crashing.

   Args:
       user_input (list): A list of user-selected features.
       fallback_price (float): The price to return if prediction fails.

   Returns:
       float: The predicted price or the fallback price.
   """
   try:
       # Attempt to create the DataFrame and then predict the price.
       user_data = pd.DataFrame([user_input], columns=x.columns)
       predicted_price = model.predict(user_data)
       return predicted_price[0]
   except ValueError:
       # If prediction fails, return the fallback price.
       return fallback_price

st.title("Group T Laptop Price Project")
st.write("Choose the laptop features to know the price")

# Creation of selection box to get user input

Company = st.selectbox( "Company" ,sorted(list(set(x["Company"].tolist()))))
Product = st.selectbox( "Product" ,sorted(list(set(x["Product"].tolist()))))
TypeName = st.selectbox( "Type" ,sorted(list(set(x["TypeName"].tolist()))))
ScreenResolution = st.selectbox( "Screen Resolution" ,sorted(list(set(x["ScreenResolution"].tolist()))))
Cpu = st.selectbox( "CPU" ,sorted(list(set(x["Cpu"].tolist()))))
Ram = st.selectbox( "RAM" ,sorted(list(set(x["Ram"].tolist()))))
Memory = st.selectbox( "Memory" ,sorted(list(set(x["Memory"].tolist()))))
Gpu = st.selectbox( "GPU" ,sorted(list(set(x["Gpu"].tolist()))))
Operating_System = st.selectbox( "Operating_System" ,sorted(list(set(x["Operating_System"].tolist()))))

# shows the price of the device
if st.button("Click"):
    user_input = [Company, Product, TypeName, ScreenResolution, Cpu, Ram, Memory, Gpu, Operating_System]
    
    # Calculate the average price to use as a fallback.
    average_price = y.mean()

    # Get the price using the updated function.
    predicted_price = get_price(user_input, average_price)
    
    # Display the result.
    text = f"The price is £{predicted_price:,.2f}"
    st.write(text)
