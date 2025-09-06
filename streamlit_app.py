import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st

# Expanded dataset to handle more user combinations.
data = {
    'Company': ['Apple', 'Dell', 'HP', 'Lenovo', 'Apple', 'Dell', 'HP', 'Acer', 'Asus', 'Microsoft', 'Dell', 'HP', 'Lenovo', 'Apple', 'Dell', 'HP', 'Lenovo', 'Acer', 'Asus', 'Microsoft'],
    'Product': ['Macbook Pro', 'XPS 13', 'Spectre x360', 'Yoga C930', 'Macbook Air', 'Inspiron', 'Envy 13', 'Swift 3', 'Zenbook', 'Surface Pro', 'Latitude', 'Pavilion', 'ThinkPad', 'iMac', 'Alienware', 'Omen', 'IdeaPad', 'Aspire', 'ROG', 'Surface Book'],
    'TypeName': ['Ultrabook', 'Ultrabook', '2 in 1 Convertible', '2 in 1 Convertible', 'Ultrabook', 'Notebook', 'Notebook', 'Ultrabook', 'Ultrabook', '2 in 1 Convertible', 'Notebook', 'Notebook', 'Notebook', 'Ultrabook', 'Gaming', 'Gaming', 'Notebook', 'Notebook', 'Gaming', 'Ultrabook'],
    'Inches': [13.3, 13.3, 13.3, 13.9, 13.3, 15.6, 13.3, 14.0, 14.0, 12.3, 14.0, 15.6, 14.0, 21.5, 17.3, 15.6, 14.0, 15.6, 15.6, 13.5],
    'ScreenResolution': ['Retina Display', '4K Ultra HD', 'Full HD', '4K Ultra HD', 'Retina Display', 'Full HD', 'Full HD', 'Full HD', 'Full HD', 'Full HD', 'Full HD', 'Full HD', 'Full HD', '4K Ultra HD', 'Full HD', 'Full HD', 'Full HD', 'Full HD', 'Full HD', '3K Display'],
    'Cpu': ['Intel Core i7', 'Intel Core i7', 'Intel Core i7', 'Intel Core i7', 'Intel Core i5', 'Intel Core i5', 'Intel Core i5', 'Intel Core i7', 'Intel Core i7', 'Intel Core i5', 'Intel Core i7', 'AMD Ryzen', 'Intel Core i7', 'Intel Core i5', 'Intel Core i7', 'Intel Core i7', 'Intel Core i5', 'AMD Ryzen', 'Intel Core i7', 'Intel Core i7'],
    'Ram': ['16GB', '8GB', '16GB', '16GB', '8GB', '8GB', '8GB', '16GB', '16GB', '8GB', '16GB', '12GB', '16GB', '8GB', '32GB', '16GB', '8GB', '8GB', '16GB', '16GB'],
    'Memory': ['256GB SSD', '512GB SSD', '1TB SSD', '1TB SSD', '128GB SSD', '1TB HDD', '512GB SSD', '256GB SSD', '512GB SSD', '256GB SSD', '512GB SSD', '256GB SSD', '1TB SSD', '256GB SSD', '1TB SSD + 1TB HDD', '512GB SSD', '1TB HDD', '256GB SSD', '1TB SSD', '512GB SSD'],
    'Gpu': ['Intel Iris Plus Graphics', 'Intel Iris Plus Graphics', 'Nvidia GeForce MX150', 'Intel UHD Graphics 620', 'Intel Iris Plus Graphics', 'AMD Radeon 520', 'Nvidia GeForce MX150', 'Intel UHD Graphics 620', 'Nvidia GeForce MX150', 'Intel UHD Graphics 620', 'Nvidia GeForce MX150', 'AMD Radeon', 'Intel UHD Graphics 620', 'Intel Iris Plus Graphics', 'Nvidia GeForce GTX 1070', 'Nvidia GeForce GTX 1060', 'AMD Radeon', 'AMD Radeon', 'Nvidia GeForce GTX 1070', 'Intel UHD Graphics 620'],
    'Operating_System': ['macOS', 'Windows 10', 'Windows 10', 'Windows 10', 'macOS', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10', 'macOS', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10', 'Windows 10'],
    'Weight': [1.37, 1.29, 1.32, 1.35, 1.25, 2.3, 1.31, 1.45, 1.2, 0.77, 1.6, 2.1, 1.5, 5.66, 3.8, 2.6, 2.2, 2.1, 2.4, 1.5],
    'Price': [1339.69, 1495.59, 1749.69, 1999.99, 1158.69, 449.69, 1200.00, 1150.00, 1300.00, 1100.00, 1550.00, 850.00, 1650.00, 1899.99, 2500.00, 2100.00, 750.00, 600.00, 2300.00, 1950.00]
}

df = pd.DataFrame(data)

# Creation And Training Models
x = df.drop(columns =["Price", "Inches", "Weight"], axis=1 ) 
y = df["Price"] # target variable

x_train, x_test, y_train, y_test = split(x, y, test_size= 0.35, random_state=0)

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
def get_price(user_input):
   """
   Predicts the price of a laptop based on user input using the trained pipeline.

   Args:
       user_input (list): A list of user-selected features.

   Returns:
       float: The predicted price of the laptop.
   """
   # Create a DataFrame from the user input. The column names must match the original `x`.
   user_data = pd.DataFrame([user_input], columns=x.columns)
   
   # Use the pre-trained pipeline to predict the price.
   # The pipeline handles all preprocessing (encoding and scaling) automatically.
   predicted_price = model.predict(user_data)
   
   # Return the first (and only) element of the prediction array.
   return predicted_price[0]


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

user_input = [Company, Product, TypeName, ScreenResolution, Cpu, Ram, Memory, Gpu, Operating_System]

# shows the price of the device
if st.button("Click"):
    try:
        text = f"The price is £{get_price(user_input):,.2f}"
        st.write(text)
    except ValueError as e:
        st.error(f"An error occurred: The selected features may not be compatible with the model. Please try different combinations.")
