import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# Load the tokenizer from the pickle file
with open('/Users/lancesanterre/pipeline_edu/notebooks/model_token/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the Keras model from the HDF5 file
model = load_model('/Users/lancesanterre/pipeline_edu/notebooks/model_token/Simple_LSTM_1000_64.keras', compile=False)

# Define the input dimension and settings
input_dim = 1000
input_length = 10  # Ensure this matches the maxlen used during training

# Initialize session state if not already done
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

# Streamlit app
st.title("Question Classification Model")
st.write("Enter a question to get the classification vector.")

# User input
user_question = st.text_input("Enter your question:", value=st.session_state.user_question)

if st.button("Predict"):
    # Tokenize and pad new data using the loaded tokenizer
    sequences = tokenizer.texts_to_sequences([user_question])
    X_new = pad_sequences(sequences, maxlen=input_length)  # Use the same maxlen as used during training

    # Make predictions
    predictions = model.predict(X_new)
    predictions_percent = np.round(predictions * 100, 2)
    predictions_hundredth = np.round(predictions, 2)

    # Create a DataFrame with column names 'WHAT', 'HOW', 'WHY'
    columns = ['WHAT', 'HOW', 'WHY']
    predictions_df = pd.DataFrame(predictions_percent, columns=columns)

    # Update session state
    st.session_state.prediction = {
        'percent': predictions_df,
        'hundredth': predictions_hundredth
    }
    st.session_state.user_question = user_question

if st.session_state.prediction:
    # Display results as a table with column names 'WHAT', 'HOW', 'WHY'
    st.write("Predictions as percentages:")
    st.table(st.session_state.prediction['percent'])


# Reset button to clear results
if st.button("Reset"):
    st.session_state.prediction = None
    st.session_state.user_question = ""
    st.experimental_rerun()  # Rerun the app to clear inputs and predictions
