from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import TextClassifierInference
import os

# Initialize FastAPI app
app = FastAPI(
    title="Ecommerce Classification API",
    description="This is a sample API with Swagger UI.",
    version="1.0.0",
    docs_url="/swagger",  # Change the Swagger UI URL
    redoc_url=None  # Disable ReDoc
)

# Check if model path exists
MODEL_PATH = "./src/saved_models"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path '{MODEL_PATH}' does not exist.")

# Load the model for inference
try:
    inference: TextClassifierInference = TextClassifierInference(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Request body schema
class TextRequest(BaseModel):
    text: str

# Define prediction endpoint
@app.post("/predict", response_model=dict[str, str])
async def predict(request: TextRequest) -> dict[str, str]:
    """Receive text input and return predicted class label."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        prediction = inference.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {"text": request.text, "prediction": prediction}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

