import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
from utils.face_detector import FaceDetector
from utils.face_embedder import FaceEmbedder
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition App")

        self.face_detector = FaceDetector()
        self.face_embedder = FaceEmbedder()
        self.known_faces = self.load_known_faces()
        self.video_source = 0  # 0 untuk webcam default, bisa diganti dengan path file video
        self.cap = cv2.VideoCapture(self.video_source)
        self.current_frame = None
        
        # UI Elements
        self.label = tk.Label(window)
        self.label.pack(padx=10, pady=10)

        self.add_face_button = tk.Button(window, text="Add Face", command=self.add_face_dialog)
        self.add_face_button.pack(side="left", padx=10, pady=10)

        self.start_recognition_button = tk.Button(window, text="Start Recognition", command=self.start_recognition)
        self.start_recognition_button.pack(side="left", padx=10, pady=10)
        
        self.stop_recognition_button = tk.Button(window, text="Stop Recognition", command=self.stop_recognition)
        self.stop_recognition_button.pack(side="left", padx=10, pady=10)

        self.is_recognizing = False

        self.update_frame()

    def load_known_faces(self):
         known_faces = {}
         data_dir = "user/data"
         os.makedirs(data_dir, exist_ok=True) # Make sure the directory exists
         for filename in os.listdir(data_dir):
             if filename.endswith(".jpg") or filename.endswith(".png"):
                try:
                     name = os.path.splitext(filename)[0]
                     image_path = os.path.join(data_dir, filename)
                     image = cv2.imread(image_path)
                     faces = self.face_detector.detect_faces(image)
                     if len(faces) > 0:
                         x, y, w, h = faces[0]
                         face = image[y:y+h, x:x+w]
                         embedding = self.face_embedder.get_embedding(face)
                         known_faces[name] = embedding
                     else:
                         print(f"No faces detected in {filename}")

                except Exception as e:
                     print(f"Error loading face in {filename}: {e}")
         return known_faces

    def add_face_dialog(self):
         name = tk.simpledialog.askstring("Add Face", "Enter the name of the person:")
         if name:
             file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
             if file_path:
                 image = cv2.imread(file_path)
                 if image is not None:
                     faces = self.face_detector.detect_faces(image)
                     if len(faces) > 0:
                         x, y, w, h = faces[0]
                         face = image[y:y+h, x:x+w]
                         embedding = self.face_embedder.get_embedding(face)
                         self.known_faces[name] = embedding
                         output_path = os.path.join("data", f"{name}.jpg")
                         cv2.imwrite(output_path, image)  # Save the original image
                         messagebox.showinfo("Success", f"Face for {name} added successfully!")
                     else:
                         messagebox.showerror("Error", "No faces detected in image.")
                 else:
                     messagebox.showerror("Error", "Failed to read the image.")

    def recognize_face(self, face_image):
        embedding = self.face_embedder.get_embedding(face_image)
        
        best_match = None
        best_similarity = 0.0
        
        for name, known_embedding in self.known_faces.items():
            similarity = cosine_similarity(embedding.reshape(1, -1), known_embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        if best_similarity > 0.5: # Threshold
            return best_match
        else:
            return "Unknown"
        
    def start_recognition(self):
         self.is_recognizing = True

    def stop_recognition(self):
        self.is_recognizing = False
    
    def update_frame(self):
         if self.cap.isOpened():
             ret, frame = self.cap.read()
             if ret:
                 self.current_frame = frame
                 if self.is_recognizing:
                     frame_copy = frame.copy() # Copy frame for processing

                     faces = self.face_detector.detect_faces(frame_copy)
                     for (x, y, w, h) in faces:
                         face_image = frame_copy[y:y+h, x:x+w]
                         name = self.recognize_face(face_image)

                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                         cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
                 image = Image.fromarray(frame)
                 photo = ImageTk.PhotoImage(image=image)
                 self.label.config(image=photo)
                 self.label.image = photo
             
         self.window.after(10, self.update_frame) # Update every 10ms

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()