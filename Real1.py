import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to convert sqft ranges like '1510 - 1670' to a single float
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df = df.dropna(subset=['size', 'bath', 'total_sqft', 'price'])

    # Create new column 'bhk'
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

    # Convert total_sqft
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

    # Drop rows where sqft conversion failed
    df = df.dropna(subset=['total_sqft'])

    return df

df = load_data()

# Prepare features and target
X = df[['total_sqft', 'bath', 'bhk']]
y = df['price']

# Split and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")
st.markdown("Enter the property details below to estimate the selling price (in Lakhs).")

# User inputs
sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0, value=1000.0, step=50.0)
bath = st.slider("Number of Bathrooms", 1, 10, 2)
bhk = st.slider("Number of Bedrooms (BHK)", 1, 10, 2)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[sqft, bath, bhk]])
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f} Lakhs")

# Optional: Show sample data
with st.expander("🔍 See sample of the dataset"):
    st.dataframe(df.head(10))
