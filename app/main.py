# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the tokenizer from the pickle file
with open('/Users/lancesanterre/pipeline_edu/notebooks/model_token/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the Keras model
model = load_model('/Users/lancesanterre/pipeline_edu/notebooks/model_token/Simple_LSTM_1000_64.keras', compile=False)

# Define input dimensions
input_length = 10  # Ensure this matches the maxlen used during training

# State management
app = FastAPI()
reset_state = {"has_reset": False}

# Define the input and output schema using Pydantic
class QuestionInput(BaseModel):
    question: str

class PredictionOutput(BaseModel):
    prediction: list  # Since prediction will be a 1x3 vector


def predict_question(question: str):
    """Predicts the classification vector for the input question."""
    # Tokenize and pad the input question
    sequences = tokenizer.texts_to_sequences([question])
    X_new = pad_sequences(sequences, maxlen=input_length)
    
    # Make prediction
    prediction = model.predict(X_new)[0]  # Extract the first vector if model returns a batch
    
    # Return prediction as a list to fit the output schema
    return prediction.tolist()


@app.get("/")
async def read_root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Question Classification API. Use /predict to classify questions."}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: QuestionInput):
    if reset_state["has_reset"]:
        raise HTTPException(status_code=400, detail="Reset before making a new prediction.")

    # Perform prediction and return as a 1x3 vector
    prediction_vector = predict_question(input_data.question)
    
    # Ensure it's a 1x3 vector; otherwise, handle errors
    if len(prediction_vector) != 3:
        raise HTTPException(status_code=400, detail="Prediction output is not a 1x3 vector.")

    return {"prediction": prediction_vector}


@app.post("/reset")
async def reset():
    """Resets the prediction state to allow for a new prediction."""
    reset_state["has_reset"] = True
    return {"message": "System reset successfully. You can now make a new prediction."}


@app.post("/confirm_reset")
async def confirm_reset():
    """Confirms the reset state and allows new predictions."""
    reset_state["has_reset"] = False
    return {"message": "Reset confirmed. You can now make new predictions."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
