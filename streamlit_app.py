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
    """
    Predicts the price of a laptop based on user input.
    
    Args:
        user_input (list): A list of user-selected features.
        
    Returns:
        float: The predicted price.
    """
    try:
        # Create DataFrame with the same structure as training data
        user_data = pd.DataFrame([user_input], columns=x.columns)
        
        # Predict the price
        predicted_price = model.predict(user_data)
        return predicted_price[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Return a more appropriate fallback based on similar configurations
        similar_configs = df[
            (df['Company'] == user_input[0]) & 
            (df['TypeName'] == user_input[2])
        ]
        
        if len(similar_configs) > 0:
            return similar_configs['Price'].mean()
        else:
            return y.mean()

st.title("Group T Laptop Price Project")
st.write("Choose the laptop features to know the price")

# Get unique values for each category, ensuring they exist in the dataset
def get_unique_sorted(column_name):
    return sorted([str(val) for val in x[column_name].unique() if pd.notna(val)])

# Creation of selection box to get user input
Company = st.selectbox("Company", get_unique_sorted("Company"))
Product = st.selectbox("Product", get_unique_sorted("Product"))
TypeName = st.selectbox("Type", get_unique_sorted("TypeName"))
ScreenResolution = st.selectbox("Screen Resolution", get_unique_sorted("ScreenResolution"))
Cpu = st.selectbox("CPU", get_unique_sorted("Cpu"))
Ram = st.selectbox("RAM", get_unique_sorted("Ram"))
Memory = st.selectbox("Memory", get_unique_sorted("Memory"))
Gpu = st.selectbox("GPU", get_unique_sorted("Gpu"))
Operating_System = st.selectbox("Operating System", get_unique_sorted("Operating_System"))

# shows the price of the device
if st.button("Predict Price"):
    user_input = [Company, Product, TypeName, ScreenResolution, Cpu, Ram, Memory, Gpu, Operating_System]
    
    # Get the price using the updated function
    predicted_price = get_price(user_input)
    
    # Display the result
    st.success(f"The estimated price is £{predicted_price:,.2f}")
    
    # Show some context - similar laptops in the dataset
    similar_laptops = df[
        (df['Company'] == Company) & 
        (df['TypeName'] == TypeName)
    ].head(3)
    
    if len(similar_laptops) > 0:
        st.write("Similar laptops in our dataset:")
        st.dataframe(similar_laptops[['Company', 'Product', 'TypeName', 'Price']])
