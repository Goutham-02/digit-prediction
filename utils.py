# utils.py
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Grayscale(),           
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def process_image(image_file):
    image = Image.open(image_file).convert("L") 
    tensor = transform(image)
    tensor = tensor.unsqueeze(0) 
    tensor = 1 - tensor
    return tensor

