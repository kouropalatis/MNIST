import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from http import HTTPStatus

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown. We load the heavy model into memory ONCE 
    and attach it to the app state so it's accessible to all routes.
    """
    print("🚀 Startup: Loading VisionEncoderDecoderModel (vit-gpt2)...")
    model_id = "nlpconnect/vit-gpt2-image-captioning"
    
    try:
        # Load the model, processor, and tokenizer
        app.state.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        app.state.feature_extractor = ViTImageProcessor.from_pretrained(model_id)
        app.state.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Move to GPU if available, otherwise CPU
        app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app.state.model.to(app.state.device)
        print(f"✅ Model loaded on {app.state.device}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError("Model loading failed")

    yield  # --- App is now running and serving requests ---

    print("🛑 Shutdown: Cleaning up GPU/RAM resources...")
    del app.state.model
    del app.state.feature_extractor
    del app.state.tokenizer

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Image Captioning Service",
    description="An API that generates text descriptions for uploaded images using ViT-GPT2.",
    lifespan=lifespan
)

# --- Routes ---

@app.get("/")
def health_check():
    """Simple endpoint to verify the API is alive."""
    return {
        "status": HTTPStatus.OK.phrase,
        "code": HTTPStatus.OK,
        "message": "Captioning Service is active"
    }

@app.post("/caption/")
async def generate_caption(
    data: UploadFile = File(...), 
    max_length: int = 16, 
    num_beams: int = 4
):
    """
    Receives an image and returns a generated caption.
    Includes optional hyperparameters for max_length and beam search.
    """
    # 1. Validate file type
    if data.content_type is None or not data.content_type.startswith("image/"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, 
            detail="File uploaded is not an image."
        )

    try:
        # 2. Open image using PIL
        i_image = Image.open(data.file)
        # Use a new variable for the RGB converted image to satisfy Mypy
        if i_image.mode != "RGB":
            processed_image = i_image.convert(mode="RGB")
        else:
            processed_image = i_image

        # 3. Pre-process image (convert to tensors)
        pixel_values = app.state.feature_extractor(
            images=[processed_image], 
            return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(app.state.device)

        # 4. Perform Inference (Generate IDs)
        output_ids = app.state.model.generate(
            pixel_values, 
            max_length=max_length, 
            num_beams=num_beams
        )

        # 5. Decode predicted IDs to strings
        preds = app.state.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        captions = [pred.strip() for pred in preds]

        return {
            "filename": data.filename,
            "captions": captions,
            "config": {
                "max_length": max_length,
                "num_beams": num_beams,
                "device": str(app.state.device)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )