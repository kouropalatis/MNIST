import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
from http import HTTPStatus
import io
import wandb
import os

from mnist.model import MyAwesomeModel

# Constants
MODEL_ARTIFACT_PATH = "s250269-danmarks-tekniske-universitet-dtu/wandb-registry-Mnist_models/corrupt_mnist_models:latest"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown.
    Downloads the latest model from WandB and loads it into memory.
    """
    print(f"🚀 Startup: Fetching model from {MODEL_ARTIFACT_PATH}...")
    
    try:
        # Initialize WandB API (assumes WANDB_API_KEY is set in env)
        api = wandb.Api()
        artifact = api.artifact(MODEL_ARTIFACT_PATH)
        artifact_dir = artifact.download(root="downloaded_model")
        model_path = os.path.join(artifact_dir, "model.pth")
        
        # Load Model
        model = MyAwesomeModel().to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        app.state.model = model
        print(f"✅ Model loaded on {DEVICE}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # We don't raise here to allow the app to start (e.g. for health checks), 
        # but prediction endpoints will fail.
        app.state.model = None

    yield

    print("🛑 Shutdown: Cleaning up resources...")
    if hasattr(app.state, "model"):
        del app.state.model

app = FastAPI(
    title="MNIST Classifier API",
    description="Classifies digit images (0-9) using a trained CNN.",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    return {
        "status": HTTPStatus.OK.phrase,
        "model_loaded": app.state.model is not None,
        "device": str(DEVICE)
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get the predicted digit.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert to grayscale and resize to 28x28 matches training data
        processed_image = image.convert("L").resize((28, 28))
    
        import torchvision.transforms.functional as F
        tensor_img = F.to_tensor(processed_image).unsqueeze(0).to(DEVICE)
        
        # Normalize (standard approximation for MNIST)
        # Verify against your data.py if possible, but this is a safe baseline
        tensor_img = (tensor_img - 0.1307) / 0.3081

        with torch.no_grad():
            outputs = app.state.model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_digit = int(torch.argmax(probabilities, dim=1).item())
            confidence = probabilities[0][predicted_digit].item()

        return {
            "prediction": predicted_digit,
            "confidence": f"{confidence:.2%}",
            "probabilities": probabilities.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))