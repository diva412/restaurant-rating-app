import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model and encoders
model = joblib.load("restaurant_rating_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

tab1, tab2 = st.tabs(["🔮 Predict Rating", "📊 EDA Dashboard"])

with tab1:
    st.title("🏨 Restaurant Rating Predictor")
    st.markdown("Predict restaurant ratings based on cuisine, pricing, delivery, and table booking options.")

    cuisine_options = label_encoders['cuisines'].classes_
    selected_cuisine = st.selectbox("🍽️ Cuisine Type", cuisine_options)

    price_range = st.slider("💰 Price Range (1 = Cheap, 4 = Expensive)", 1, 4)

    online_options = label_encoders['online_order'].classes_
    online_order = st.selectbox("📦 Online Order Available?", online_options)

    booking_options = label_encoders['book_table'].classes_
    book_table = st.selectbox("🪑 Table Booking Available?", booking_options)

    if st.button("Predict Rating"):
        cuisine_encoded = label_encoders['cuisines'].transform([selected_cuisine])[0]
        online_encoded = label_encoders['online_order'].transform([online_order])[0]
        booking_encoded = label_encoders['book_table'].transform([book_table])[0]

        input_features = np.array([[cuisine_encoded, price_range, online_encoded, booking_encoded]])
        predicted_rating = model.predict(input_features)[0]

        st.success(f"⭐ Predicted Restaurant Rating: {round(predicted_rating, 2)} / 5")

with tab2:
    st.header("📊 EDA Dashboard")

    df = pd.read_csv("zomato.csv")
    df = df[['cuisines', 'price_range', 'online_order', 'book_table', 'aggregate_rating']].dropna()

    st.write("Sample Data", df.head())

    st.subheader("Rating Distribution by Price Range")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="price_range", y="aggregate_rating", estimator=np.mean, ax=ax)
    st.pyplot(fig)

