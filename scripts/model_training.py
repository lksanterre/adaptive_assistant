from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, Dropout
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def build_model(input_shape):
    """Builds a simple LSTM-based neural network model."""
    model = Sequential([
        Embedding(input_dim=1000, output_dim=128, input_length=input_shape),
        LSTM(64, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(filtered_questions, filtered_labels, model, input_dim, input_length, output_dim):
    """Train the model using the provided data and model."""

    tokenizer = Tokenizer(num_words=input_dim)  
    tokenizer.fit_on_texts(filtered_questions)
    sequences = tokenizer.texts_to_sequences(filtered_questions)
    X = pad_sequences(sequences, maxlen=input_length)

    # Convert labels to numpy array
    y = np.array(filtered_labels.tolist())

    # Model: Simple LSTM
    model_name = f"Simple_LSTM_{input_dim}_{output_dim}"

    # Compile and train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate and log
    loss, accuracy = model.evaluate(X_test, y_test)
    return model_name, model, loss, accuracy