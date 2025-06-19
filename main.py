from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
from model import CNN
from utils import process_image
from gradcam import apply_gradcam, overlay_heatmap_on_image, encode_image_to_base64

app = FastAPI()

# Enable CORS for all origins (or restrict it to specific ones)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend origin (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Preprocess the uploaded image
        image_tensor = process_image(file.file)  # Should return a tensor of shape [1, 1, 28, 28]

        # Run inference
        output = model(image_tensor)
        predicted_class = torch.argmax(output, 1).item()

        # Grad-CAM and heatmap
        heatmap = apply_gradcam(model, image_tensor, predicted_class)
        overlay_image = overlay_heatmap_on_image(heatmap, image_tensor)
        heatmap_base64 = encode_image_to_base64(overlay_image)

        # Return the response
        return JSONResponse(content={
            "prediction": predicted_class,
            "heatmap": heatmap_base64
        })

    except Exception as e:
        # Log and return error if anything fails
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
