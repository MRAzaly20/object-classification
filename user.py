import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

# Fungsi untuk memuat model
def load_model(model_path, num_classes, device):
    model = models.mobilenet_v2(pretrained=False)  # Jangan menggunakan parameter pretrained = True
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Fungsi untuk melakukan prediksi dari array gambar
def predict_face_from_array(model, image_array, train_dataset, device, transform, threshold, reference_embedding=None):
    try:
        model.eval()
        img = Image.fromarray(image_array).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            output_np = output.cpu().detach().numpy()
            if len(train_dataset.classes) > 1:
              probabilities = np.exp(output_np) / np.sum(np.exp(output_np))
              predicted_class = np.argmax(probabilities)
              predicted_probability = probabilities[0][predicted_class]
              if predicted_probability > threshold:
                 return train_dataset.classes[predicted_class]
              else:
                 return "User Not Recognized"
            else: # jika hanya satu kelas maka output adalah feature embedding
              if reference_embedding is not None:
                distance = np.linalg.norm(output_np - reference_embedding) # euclidean distance
                if distance > threshold:
                    return  train_dataset.classes[0] + str(distance)
                else:
                    return "User Not Recognized" +  str(distance)
              else:
                return output_np
    except Exception as e:
        print(f"Error during prediction from array: {e}")
        return None
        
def predict_face(model, image_path, train_dataset, device, transform):
  try:
      model.eval()
      img = Image.open(image_path).convert('RGB') #convert to RGB in case image is grayscale
      img_tensor = transform(img).unsqueeze(0)
      img_tensor = img_tensor.to(device)
      with torch.no_grad():
          output = model(img_tensor)
          _, predicted_class = torch.max(output, 1)
          return train_dataset.classes[predicted_class.item()] #return class_name
  except Exception as e:
      print(f"Error during prediction: {e}")
      return None