import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st

# Defining dataset to a variable
df = pd.read_csv("cleaned_laptop_price_dataset.csv")

# FIX: Drop any unnamed columns and strip whitespace from column names
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

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

# Function to predict laptop price
def get_price(user_input):

   new_data = {new_cols :[user_input[a]] for new_cols, a in zip(x.columns, range(len(x.columns)))}
   
   new_df = pd.DataFrame(new_data)
  
   # New data to numeric value

   onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

   for cols in new_df.select_dtypes(include ="object").columns:
      
      encoded_features = onehot_encoder.fit_transform(new_df[[cols]])
      encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out([cols]))
      # Join with the original DataFrame
      df_encoded = pd.concat([new_df, encoded_df], axis=1)

   return model.predict(df_encoded)


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
    text = f"The price is £{get_price(user_input)[0]:,.2f}"
    st.write(text)
    
