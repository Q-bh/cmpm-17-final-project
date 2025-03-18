# Excess libraries are included in case they are needed
# They will be removed by the final project's completion if they remain unused
from PIL import Image
import torch
from torchvision.transforms import v2

from main import CNN_Main

model_save_path = "cnn_model.pt"

loaded_model = CNN_Main()
loaded_model.load_state_dict(torch.load(model_save_path, weights_only=True))
loaded_model.eval()
print(f"Model loaded from {model_save_path}")

transforms = v2.Compose([
        v2.Resize((128, 128)),
        v2.Grayscale(1),
        v2.ToTensor(),
        v2.Normalize([0.5], [0.5])
    ])

# CHANGE FILE NAME HERE: "./demo/The_<EMOTION>_Rock.png"
image_path = "./demo/The_Anger_Rock.png"

image = Image.open(image_path)

image_tensor = transforms(image).unsqueeze(0)

with torch.no_grad():
    pred = loaded_model(image_tensor)
    confidences = torch.softmax(pred, dim=1)
    confidence, predicted_class = torch.max(confidences, dim=1)

emotions_list = ["anger", "fear", "happy", "neutral", "sad", "surprise"]

# Convert confidences to percent probabilities
confidence_list = list(map(lambda confidence: confidence * 100, confidences.squeeze().tolist()))

print(f"""
The model thinks...
anger: {confidence_list[0]:.2f}%
fear: {confidence_list[1]:.2f}%
happy: {confidence_list[2]:.2f}%
neutral: {confidence_list[3]:.2f}%
sad: {confidence_list[4]:.2f}%
surprise: {confidence_list[5]:.2f}%
""")
print(f"Predicted class: {emotions_list[predicted_class.item()]}, Confidence: {confidence.item() * 100:.2f}%")