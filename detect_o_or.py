import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


tf.config.run_functions_eagerly(True)

#from google.colab import drive
#drive.mount('/content/drive')

train_path = os.path.abspath("datasets/archive/test/training/")  # Absolute Path
print("Absolute train path:", train_path)

def get_model(IMG_SIZE):
    model = Sequential([
        # Model layers
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#store keselurhan predictor
main_pred = []
error = []

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 100
N_SPLIT = 5

#menetapkan nilai per-foldnya
acc_per_fold = []
loss_per_fold = []


#inisialisasi generator data
datagen = ImageDataGenerator(rescale=1. / 255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

#inisialisasi k-fold
data_gen = datagen.flow_from_directory(train_path,
                                       target_size=(IMG_SIZE, IMG_SIZE),
                                       batch_size=BATCH_SIZE,
                                       class_mode="binary",
                                       shuffle=True)
print("Class Indices:", data_gen.class_indices)
#Debugging: looping throught a few batches
for i, (data_batch, label_batch) in enumerate(data_gen):
       if i >= 2:
          break
       print("Data batch shape:", data_batch.shape)
       print("Labels shape:", label_batch.shape)
       print("Label Example",label_batch[0])
       print("Data Example", data_batch[0])


# Check data shape and labels (keep)
data, labels=next(data_gen)
print("Shape of a single batch of data:", data.shape)
print("Shape of a single batch of labels:", labels.shape)
print("Example of data:", data[0])  # Print the first image
print("Example of labels:", labels[0]) # Print the first label
print("Total samples:", data_gen.n)
print("Number of class indices:", len(data_gen.class_indices))
print("Class indices:", data_gen.class_indices)

# kfold = StratifiedKFold(n_splits=N_SPLIT,
#  shuffle=True,
#  random_state=42)


#variable menghitung setiap pembagiaannya
j = 0
model = get_model(IMG_SIZE)

# Assuming you'd want a train/valid split like this from the generated data
x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.2, random_state=42) # Adjust test_size as needed
validation_steps = len(x_valid) // BATCH_SIZE
history = model.fit(data_gen,  # Correctly passing the generator
                    epochs=EPOCHS,
                    validation_data=(x_valid, y_valid),
                    steps_per_epoch=data_gen.n // BATCH_SIZE,
                    validation_steps=validation_steps)


scores = model.evaluate(x_valid, y_valid, verbose=0)
pred = model.predict(x_valid)
y_pred = (pred > 0.5).astype(int).flatten()

cf_matrix = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:\n", cf_matrix)
print(f"Accuracy Score: {accuracy_score(y_valid,y_pred)}")

model.save(f'/content/drive/MyDrive/deteksi-gambar/model/model{j}.h5')
print(f'Score for fold {j}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
acc_per_fold.append(scores[1] * 100)
loss_per_fold.append(scores[0])

print("TensorFlow Version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))