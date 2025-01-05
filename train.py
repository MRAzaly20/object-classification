import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import numpy as np

# 1. Persiapan Dataset
data_dir = "datasets/archive/Train_Sets" # Ganti dengan path dataset Anda
image_height = 224 # Ukuran default mobilenetv2
image_width = 224
batch_size = 32
num_epochs = 10
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transformasi gambar
train_transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data loader
train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)

# 2. Load Model MobileNetV2
model = models.mobilenet_v2(pretrained=True)
num_features = model.classifier[1].in_features # dapatkan jumlah fitur input di layer klasifikasi
model.classifier[1] = nn.Linear(num_features, num_classes) #Ganti layer klasifikasi
model = model.to(device)

# 3. Loss Function dan Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Latih Model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
      for data in val_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')


# 5. Simpan Model
torch.save(model.state_dict(), 'rev_object_recognition_mobilenetv2.pth') #Save dalam format .pth

# 6. Face Recognition (Inference - Contoh Sederhana)
def predict_face(image_path):
    model.eval()
    img = Image.open(image_path)
    img_tensor = val_transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
        return train_dataset.classes[predicted_class.item()] #return nama class yang diprediksi

# Contoh menggunakan fungsi predict_face
# Ganti "path/to/your/test_image.jpg" dengan path gambar yang ingin Anda prediksi
#test_image_path = "test/azaly.jpg"
#predicted_name = predict_face(test_image_path)
#print(f"Predicted user: {predicted_name}")