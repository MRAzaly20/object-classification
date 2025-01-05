from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import torch

class FaceEmbedder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def get_embedding(self, face_image):
        face_image = Image.fromarray(face_image)  # Konversi NumPy array ke PIL Image
        face_image = face_image.resize((160, 160)) # Resize ke 160x160
        face_image = np.array(face_image)
        face_image = (face_image - 127.5) / 128.0 # Normalisasi
        face_image = np.transpose(face_image, (2, 0, 1)) # Konversi ke format channel first
        face_image = torch.from_numpy(face_image).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(face_image).squeeze(0).cpu().numpy()
        return embedding