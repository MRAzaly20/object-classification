import torch
from torchvision import transforms, datasets
import os
from user import load_model, predict_face

# Data Path
data_dir = "user/data" # ganti path
image_height = 224
image_width = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'face_recognition_mobilenetv2.pth'


# Transform
val_transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Dataset
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=val_transform) #gunakan dataset train untuk class name

# Load Model
loaded_model = load_model(model_path, len(train_dataset.classes), device)

# Prediction
test_image_path = "test/emak.jpg"
predicted_name = predict_face(loaded_model, test_image_path, train_dataset, device, val_transform)
if predicted_name is not None:
    print(f"Predicted user: {predicted_name}")