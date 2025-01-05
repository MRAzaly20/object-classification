import cv2
import tkinter as tk
from PIL import ImageTk, Image
import torch
from torchvision import transforms, datasets
import os
import numpy as np
from user import load_model, predict_face_from_array #import prediksi dari array

root = tk.Tk()
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
lmain = tk.Label(root)
lmain.grid()

# Initialize the camera with index 1 (front camera)
cap = cv2.VideoCapture(1)

# Check that we have camera access
if not cap.isOpened():
  # Try camera index 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
      lmain.config(text="Unable to open camera. Please ensure your device supports Camera NDK API.", wraplength=lmain.winfo_screenwidth())
      root.mainloop()
else:
    # You can set the desired resolution here
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Load Model dan Path ---
data_dir = "user/data" # ganti path
image_height = 224
image_width = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'rev_object_recognition_mobilenetv2.pth'
threshold = 0.3 # ubah threshold untuk face verification, nilai kecil jika tingkat keyakinan tinggi

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

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

prev_face_frame = None
motion_threshold = 10  # Atur sesuai kebutuhan
reference_embedding = None # ubah jika punya embedding referensi untuk face verification


def refresh():
    global imgtk, prev_face_frame, reference_embedding
    ret, frame = cap.read()
    if not ret:
        # Error capturing frame, try next time
        lmain.after(0, refresh)
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # crop the face image

        # Anti-Spoofing check
        is_real = True
        if prev_face_frame is not None:
            # Resize prev_face_frame to match face_img size
            prev_face_frame_resized = cv2.resize(prev_face_frame, (face_img.shape[1], face_img.shape[0]))
            diff = cv2.absdiff(cv2.cvtColor(prev_face_frame_resized, cv2.COLOR_RGB2GRAY), cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY))
            avg_diff = np.mean(diff)
            if avg_diff < motion_threshold:
                is_real = False
                predicted_name = "Spoof Detected" # set spoof
            else:
              predicted_name = predict_face_from_array(loaded_model, face_img, train_dataset, device, val_transform, threshold=threshold, reference_embedding=reference_embedding)

        else:
            predicted_name = predict_face_from_array(loaded_model, face_img, train_dataset, device, val_transform, threshold=threshold, reference_embedding=reference_embedding)
            if reference_embedding is None:
              output_first_time = predict_face_from_array(loaded_model, face_img, train_dataset, device, val_transform, threshold=threshold, reference_embedding=None) # mendapatkan feature embedding jika reference embedding tidak ada.
              if isinstance(output_first_time, np.ndarray):
                reference_embedding = output_first_time # simpan feature embedding pertama sebagai acuan.
                predicted_name = train_dataset.classes[0] #ambil nama class
            
        if is_real:
          if isinstance(predicted_name, str):
               # Draw rect & label
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
          else:
               #jika bukan string (output feature embedding) maka jangan tampilkan text
              cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              cv2.putText(frame, str(predicted_name), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for spoof
          cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Save current frame
        prev_face_frame = face_img

    # Convert back to RGB and display
    w = lmain.winfo_screenwidth()
    h = lmain.winfo_screenheight()
    cw = frame.shape[0]
    ch = frame.shape[1]
    # In portrait, image is rotated
    cw, ch = ch, cw
    if (w > h) != (cw > ch):
        # In landscape, we have to rotate it
        cw, ch = ch, cw
        # Note that image can be upside-down, then use clockwise rotation
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Keep aspect ratio
    w = min(cw * h / ch, w)
    h = min(ch * w / cw, h)
    w, h = int(w), int(h)
    # Resize to fill the whole screen
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.configure(image=imgtk)
    lmain.update()
    lmain.after(0, refresh)

refresh()
root.mainloop()