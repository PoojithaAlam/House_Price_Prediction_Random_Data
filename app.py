import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------- Generate Fake House Data ----------
def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 500, n_samples)
    price = 50 * size + np.random.normal(0, 50000, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

# ---------- Train Model ----------
def train_model():
    df = generate_house_data(n_samples=100)
    x = df[['size']]
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model, df

# ---------- Streamlit UI ----------
st.title("🏡 House Price Predictor")
st.write("This app predicts house prices based on their size (sq. ft). The data is randomly generated.")

# Train the model
model, df = train_model()

# Show a plot of the data
st.subheader("📊 Generated Data")
fig = px.scatter(df, x='size', y='price', title='Randomly Generated House Data')
st.plotly_chart(fig)

# Get user input
st.subheader("🔢 Enter House Size")
input_size = st.number_input("House size (sq. ft)", min_value=300, max_value=5000, value=1500)

# Make prediction
if st.button("Predict Price"):
    predicted_price = model.predict([[input_size]])[0]
    st.success(f"Estimated House Price: ${predicted_price:,.2f}")
