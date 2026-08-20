import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import WasteClassifier


def predict_single_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("model.pth", map_location=device)
    classes = checkpoint['classes']

    model = WasteClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()

    print(f"Image: {image_path} -> Prediction: {classes[class_index]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Error: Please provide the image path. Example: python predict.py my_image.jpg")
    else:
        predict_single_image(sys.argv[1])
