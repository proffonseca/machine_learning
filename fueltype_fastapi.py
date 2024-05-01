from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the trained Random Forest model
model = joblib.load('fossil_electric_rf_model.joblib')

# Load the TF-IDF vectorizer
vectorizer = joblib.load('fossil_electric_vectorizer.joblib')

# Define the FastAPI app
app = FastAPI()

# Create request body model
class InputText(BaseModel):
    text: str

# Define endpoint for making predictions
@app.post("/predict/")
async def predict(input_data: InputText):
    # Vectorize the input text
    text_vectorized = vectorizer.transform([input_data.text])
    
    # Make prediction using the trained model
    prediction = model.predict(text_vectorized)
    
    # Convert prediction to human-readable output
    if prediction[0] == 1:
        result = "Electric"
    else:
        result = "Fossil"
    
    # Return the prediction
    return {"prediction": result}
