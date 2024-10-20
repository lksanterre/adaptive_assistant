from metaflow import FlowSpec, step, Parameter, kubernetes

class TrainingFlow(FlowSpec):

    # Parameters for the flow
    data_path = Parameter('data_path', default='/Users/lancesanterre/pipeline_edu/data/processed/pipeline_and_data.pkl', help="Path to the new data for training")
    input_dim = Parameter('input_dim', default=1000, help="Vocabulary size for the tokenizer")
    output_dim = Parameter('output_dim', default=64, help="Output dimension for the Embedding layer")
    input_length = Parameter('input_length', default=10, help="Input length for padding sequences")

    @step
    def start(self):
        """Install required packages and load data."""
        import os
        print("Installing required packages...")
        os.system(
            "pip install tensorflow==2.17.0 keras==3.5.0 numpy==1.26.4 "
            "pandas==2.1.4 scikit-learn==1.2.2 transformers==4.44.2 "
            "h5py==3.11.0 six==1.16.0 protobuf==4.24.4 gast==0.5.4 "
            "typing-extensions==4.12.2"
        )
        import numpy as np
        print("Packages installed successfully.")
        from data_processing import process_data
        print("Starting the training flow...")
        self.data = process_data(self.data_path)
        self.next(self.tokenize_and_transform)

    @step
    def tokenize_and_transform(self):
        """Tokenize and transform the data."""
        import numpy as np
        import os
        import pickle  # Make sure to import pickle here
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        questions = self.data[0]  # Adjust key based on your data structure
        labels = self.data[1]     # Adjust key based on your data structure

        self.tokenizer = Tokenizer(num_words=self.input_dim)
        self.tokenizer.fit_on_texts(questions)
        sequences = self.tokenizer.texts_to_sequences(questions)
        self.X = pad_sequences(sequences, maxlen=self.input_length)
        self.y = np.array(labels.tolist())  # Adjust if needed
        # Save the tokenizer to a specific location
        tokenizer_path = "//Users/lancesanterre/pipeline_edu/model_token/tokenizer.pkl"  # Specify your desired path
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True) 
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)

        print("Tokenizer saved as 'tokenizer.pkl'")
        self.next(self.train_model)
    @kubernetes
    @step
    def train_model(self):
        """Train the model on the entire dataset."""
        import os
        print("Installing required packages...")
        os.system(
            "pip install tensorflow==2.17.0 keras==3.5.0 numpy==1.26.4 "
            "pandas==2.1.4 scikit-learn==1.2.2 transformers==4.44.2 "
            "h5py==3.11.0 six==1.16.0 protobuf==4.24.4 gast==0.5.4 "
            "typing-extensions==4.12.2"
        )
        print("Training the model...")
        import numpy as np
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
        self.model = Sequential([
            Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length),
            LSTM(self.output_dim),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X, self.y, epochs=10, verbose=2)  # Adjust epochs as needed
        self.next(self.save_predictions)

    @step
    def save_predictions(self):
        """Feed the model input to the trained model, predict, and save the predictions."""
        import numpy as np
        import pickle
        predictions = self.model.predict(self.X)
        prediction_path = '/Users/lancesanterre/pipeline_edu/data/predictions/predictions.pkl'
        with open(prediction_path, 'wb') as f:
            pickle.dump(predictions, f)
            model_path = "/Users/lancesanterre/pipeline_edu/model_token"  # Specify your desired path
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        print(f"Predictions saved as '{prediction_path}'")
        self.next(self.end)

    @step
    def end(self):
        """End step."""
        print("Training flow completed!")

if __name__ == '__main__':
    TrainingFlow()
