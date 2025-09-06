import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Hardcoding a sample of the dataset to make the script self-contained.
# This replaces the need for a local "cleaned_laptop_price_dataset.csv" file.
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
x = df.drop(columns=["Price", "Inches", "Weight"], axis=1) 
y = df["Price"] # target variable

# Splitting data for training and testing
x_train, x_test, y_train, y_test = split(x, y, test_size=0.35, random_state=0)

# Identify categorical & numeric columns
categorical_columns = x.select_dtypes(include=['object']).columns
numerical_columns = x.select_dtypes(exclude=['object']).columns

# Create the ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('category', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('numbers', StandardScaler(), numerical_columns)
    ]
)

# Create the final pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model with the training data
model.fit(x_train, y_train)

# --- Function to predict laptop price ---
def get_price(user_input_list):
    """
    Predicts the price of a laptop based on user input using the trained pipeline.

    Args:
        user_input_list (list): A list of user-selected features in the correct order.

    Returns:
        float: The predicted price of the laptop.
    """
    # Create a DataFrame from the user input.
    # The columns must match the order of the training data.
    user_data = pd.DataFrame([user_input_list], columns=x.columns)
    
    # Use the pre-trained pipeline to predict the price.
    # The pipeline handles all preprocessing (encoding and scaling) automatically.
    predicted_price = model.predict(user_data)
    
    # Return the first (and only) element of the prediction array.
    return predicted_price[0]

# --- Command-line Interface (CLI) for User Input ---
def main():
    """
    A simple command-line interface to get user input and predict the price.
    """
    print("Welcome to the Laptop Price Predictor!")
    print("Please choose the laptop features to get the price.")
    
    # Get user input for each feature.
    # The list of unique values is displayed for the user.
    Company = input(f"Enter Company ({sorted(list(set(x['Company'])))}) : ")
    Product = input(f"Enter Product ({sorted(list(set(x['Product'])))}) : ")
    TypeName = input(f"Enter Type ({sorted(list(set(x['TypeName'])))}) : ")
    ScreenResolution = input(f"Enter Screen Resolution ({sorted(list(set(x['ScreenResolution'])))}) : ")
    Cpu = input(f"Enter CPU ({sorted(list(set(x['Cpu'])))}) : ")
    Ram = input(f"Enter RAM ({sorted(list(set(x['Ram'])))}) : ")
    Memory = input(f"Enter Memory ({sorted(list(set(x['Memory'])))}) : ")
    Gpu = input(f"Enter GPU ({sorted(list(set(x['Gpu'])))}) : ")
    Operating_System = input(f"Enter Operating System ({sorted(list(set(x['Operating_System'])))}) : ")

st.title("Group T Laptop Price Project")
st.write("Choose the laptop features to know the price")

    # Create the user input list in the correct order
    user_input = [Company, Product, TypeName, ScreenResolution, Cpu, Ram, Memory, Gpu, Operating_System]

    # Predict the price and format the output
    try:
        predicted_price = get_price(user_input)
        print(f"\nThe predicted price for this laptop is: £{predicted_price:,.2f}")
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        print("Please ensure your input values exactly match the examples provided.")

# Run the main function
if __name__ == "__main__":
    main()
