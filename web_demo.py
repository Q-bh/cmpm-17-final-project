import gradio as gr
from PIL import Image
import torch
from torchvision.transforms import v2
from main import CNN_Main  # Check the path

# 모델 로드
model_save_path = "cnn_model.pt"
loaded_model = CNN_Main()
loaded_model.load_state_dict(torch.load(model_save_path, weights_only=True))
loaded_model.eval()

# 전처리 파이프라인
transforms = v2.Compose([
    v2.Resize((128, 128)),
    v2.Grayscale(1),
    v2.ToTensor(),
    v2.Normalize([0.5], [0.5])
])

emotions_list = ["anger", "fear", "happy", "neutral", "sad", "surprise"]

def predict(image):
    # Image coming from Gradio is numpy array.
    image = Image.fromarray(image)
    image_tensor = transforms(image).unsqueeze(0)
    with torch.no_grad():
        pred = loaded_model(image_tensor)
        confidences = torch.softmax(pred, dim=1)
        main_confidence, main_predicted_class = torch.max(confidences, dim=1)
    # Return the result as strings
    result_text = f"Predicted: {emotions_list[main_predicted_class.item()]} (Confidence: {main_confidence.item() * 100:.2f}%)\n"
    result_text += "\n".join([f"{emotions_list[i]}: {conf.item() * 100:.2f}%" for i, conf in enumerate(confidences.squeeze())])
    return result_text

interface = gr.Interface(fn=predict, 
                        inputs=gr.Image(type="numpy", label="Upload an Image"), 
                        outputs="text",
                        title="Facial Expression classifier",
                        description="Upload an facial image to detect the emotion using our CNN model (Classes are Angry, Fear, Happy, Neutral, Sad, Surprised)")

interface.launch(share=True)