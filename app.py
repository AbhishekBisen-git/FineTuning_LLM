from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

# 1. Initialize API
app = FastAPI(title="Price Predictor LLM API", version="1.0")

# 2. Global Model Storage
model = None
tokenizer = None

# 3. Configuration (Must match training!)
MODEL_PATH = "./my_fine_tuned_price_model" # Points to your saved folder
MAX_SEQ_LENGTH = 2048

class ProductRequest(BaseModel):
    description: str

@app.on_event("startup")
async def load_model():
    """Loads model on server startup so we don't reload for every request."""
    global model, tokenizer
    print("Loading AI Model... this may take a minute.")

    # Load using Unsloth for speed (requires GPU)
    # If using CPU only, use standard transformers (slower)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    print("✅ Model Loaded Successfully!")

@app.post("/predict")
async def predict_price(request: ProductRequest):
    """The endpoint that accepts a product description and returns a price."""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Define the exact prompt format used in training
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert price predictor. Estimate the price of the following item.

### Input:
{}

### Response:
"""
    # Prepare Input
    inputs = tokenizer(
        [alpaca_prompt.format(request.description)],
        return_tensors = "pt"
    ).to("cuda")

    # Generate Prediction
    outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True)
    decoded_output = tokenizer.batch_decode(outputs)[0]

    # Clean String to extract just the price
    try:
        predicted_price = decoded_output.split("### Response:")[-1].replace("<|end_of_text|>", "").strip()
    except:
        predicted_price = "Error parsing prediction"

    return {
        "input_product": request.description,
        "predicted_price": predicted_price
    }

# To run locally: uvicorn app:app --reload