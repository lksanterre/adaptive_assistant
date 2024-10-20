from metaflow import FlowSpec, step, Parameter
import mlflow
from data_processing import process_data  
from model_training import build_model, train_model 



class TrainingFlow(FlowSpec):

    # Parameters for the flow
    seed = Parameter('seed', default=42, help="Random seed for reproducibility")

    @step
    def start(self):
        """Start step: Load and preprocess the data."""
        print("Starting the training flow...")

        # Load and preprocess data
        self.filtered_questions, self.filtered_labels = process_data('/Users/lancesanterre/pipeline_edu/data/processed/pipeline_and_data.pkl')
        
        # Define input parameters for the model
        self.input_dim = 1000  
        self.input_length = 128  
        self.output_dim = 3  # what, how, why

        self.next(self.build_model)

    @step
    def build_model(self):
        """Build the model before training."""
        print("Building the model...")

        # Define input shape as input_length
        self.input_shape = self.input_length  
        self.model = build_model(self.input_shape)
        
        self.next(self.train)

    @step
    def train(self):
        """Model training step."""
        print("Training the model...")

        # Train the model using filtered questions and labels
        self.model_name, self.model, self.loss, self.accuracy = train_model(
            self.filtered_questions,
            self.filtered_labels,
            self.model,
            self.input_dim,
            self.input_length,
            self.output_dim
        )

        # Log best parameters
        self.best_params = {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "input_length": self.input_length,
            "output_dim": self.output_dim,
            "loss": self.loss,
            "accuracy": self.accuracy,
        }
        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the trained model in MLflow."""
        print("Registering the model in MLflow...")
        mlflow.set_tracking_uri('https://mlflow-service-1073438601911.us-west2.run.app') 
        mlflow.set_experiment('model_traning_metaflow')
        
        # Start a new MLflow run
        with mlflow.start_run() as run:
            mlflow.log_params(self.best_params)
            mlflow.keras.log_model(self.model, "model")
                    # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "model_1")
        
        self.next(self.end)


    @step
    def end(self):
        """End step."""
        print("Training flow completed!")

if __name__ == '__main__':
    TrainingFlow()
