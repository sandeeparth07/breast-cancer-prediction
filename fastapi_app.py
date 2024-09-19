from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("breast_cancer_logistic_regression.pkl")

# Load the dataset for column information
data = pd.read_csv("breast-cancer.csv")

# Define the features used in the model
features = [col for col in data.columns if col not in ['id', 'diagnosis']]

# Define input schema using Pydantic
class CancerPredictionInput(BaseModel):
    feature_values: dict

# Initialize the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict/")
async def predict(input_data: CancerPredictionInput):
    # Initialize the user input with the mean values for all features
    default_input = {feature: float(data[feature].mean()) for feature in features}

    # Update default_input with the provided feature values
    default_input.update(input_data.feature_values)
    
    # Convert the input dictionary to a NumPy array in the correct order of features
    try:
        user_input = np.array([default_input[feature] for feature in features]).reshape(1, -1)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in request: {str(e)}")
    
    # Predict using the model
    prediction = model.predict(user_input)[0]
    
    # Return the prediction result
    cancer_type = "Malignant" if prediction == 1 else "Benign"
    return {"predicted_cancer_type": cancer_type}


@app.on_event("startup")
async def startup_event():
    print("FastAPI is running! Welcome to the server.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Breast Cancer Prediction API"}
