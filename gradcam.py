import cv2
import torch
import numpy as np
import io
import base64
from PIL import Image

def apply_gradcam(model, image_tensor, predicted_class):
    model.eval()
    output = model(image_tensor)
    model.zero_grad()

    score = output[0, predicted_class]
    score.backward()

    gradients = model.gradients[0]      # shape: [C, H, W]
    activations = model.features[0]     # shape: [C, H, W]

    pooled_gradients = torch.mean(gradients, dim=(1, 2))

    for i in range(activations.shape[0]):
        activations[i] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize to match image size
    heatmap = cv2.resize(heatmap, (28, 28))
    return heatmap

def overlay_heatmap_on_image(heatmap, original_tensor):
    original_image = original_tensor.squeeze().cpu().numpy()
    original_image = (original_image * 255).astype(np.uint8)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

    return overlay

def encode_image_to_base64(img_np):
    img_pil = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
