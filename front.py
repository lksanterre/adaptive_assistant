# frontend.py
import streamlit as st
import requests

# Streamlit UI Configuration
st.set_page_config(page_title="Question Classification", layout="centered")
st.title("Question Classification Model")
st.write("Intern Question to Vect")

# User Input
question = st.text_input("Enter your question:")

# API URL
api_url = "http://127.0.0.1:8000/predict"  # Ensure this matches your FastAPI URL

# Predict Button
if st.button("Predict"):
    if question.strip() == "":
        st.error("Please enter a valid question.")
    else:
        # Send request to FastAPI backend
        response = requests.post(api_url, json={"question": question})

        # Display the prediction results
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.success("Prediction received!")
            st.write(f"Prediction Vector: {prediction}")
        else:
            st.error("Failed to get a prediction. Please try again.")


