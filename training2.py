import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tensorflow as tf
#from google.colab import drive
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split
from imblearn.under_sampling import RandomUnderSampler


tf.config.run_functions_eagerly(True)

#from google.colab import drive
#drive.mount("/content/drive")

TRAIN_PATH = "datasets/test/training/"
     

for class_name in ["electric_bus", "electric_car"]:
    class_path = os.path.join(TRAIN_PATH, class_name)
    filenames = os.listdir(class_path)[:10]
    print(f"Class: {class_name}, Filenames: {filenames}")

def get_model(IMG_SIZE):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(2, activation="softmax")
  ])

  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  return model
  model.summary()
     
#store keselurhan predictor
main_pred = []
error = []

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 100
N_SPLIT = 5
N_REPEAT = 5

#menetapkan nilai per-foldnya
acc_per_fold = []
loss_per_fold = []


datagen_train = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

datagen_validation = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
    )
kfold = RepeatedKFold(n_splits=N_SPLIT, n_repeats=N_REPEAT, random_state=42)
j = 0

electric_bus_files = os.listdir(os.path.join(TRAIN_PATH, "electric_bus"))
electric_car_files = os.listdir(os.path.join(TRAIN_PATH, "electric_car"))

num_samples_per_class = min(len(electric_bus_files), len(electric_car_files))

np.random.shuffle(electric_bus_files)
np.random.shuffle(electric_car_files)

electric_bus_files = electric_bus_files[:num_samples_per_class]
electric_car_files = electric_car_files[:num_samples_per_class]

all_files = electric_bus_files + electric_car_files
all_labels = ["electric_bus"] * len(electric_bus_files) + ["electric_car"] * len(electric_car_files)

for train_idx, valid_idx in kfold.split(all_files, all_labels):
    j += 1

    x_train_filenames = np.array(all_files)[train_idx]
    y_train_labels = np.array(all_labels)[train_idx]
    x_valid_filenames = np.array(all_files)[valid_idx]
    y_valid_labels = np.array(all_labels)[valid_idx]

    training_set = datagen_train.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        subset="training"
    )

    validation_set = datagen_validation.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        subset="validation"
    )

    model = get_model(IMG_SIZE)

    history = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=EPOCHS,
        steps_per_epoch=len(training_set) // BATCH_SIZE,
        validation_steps=len(validation_set) // BATCH_SIZE,
        verbose = 1
    )

    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.title(f"Model Accuracy and Loss Fold {j}")
    plt.ylabel("value")
    plt.xlabel("No. Epoch")
    plt.legend(loc="upper left")
    plt.show()

    scores = model.evaluate(validation_set, verbose=0)

    pred = model.predict(validation_set)
    y_pred = np.argmax(pred, axis=1)

    cf_matrix = confusion_matrix(validation_set.classes, y_pred)
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(cf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.xlabel("Prediction")
    plt.ylabel("True")
    ax.xaxis.set_ticklabels(["electric_bus", "electric_car"])
    ax.yaxis.set_ticklabels(["electric_bus", "electric_car"])
    plt.show()

    model.save(f"model/model_v3/model{j}.h5")
    print(f"Score for fold {j}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%")
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

plt.plot(acc_per_fold, label="Accuracy")
plt.plot(loss_per_fold, label="Loss")
plt.title("K-Fold Accuracy and Loss")
plt.ylabel("Value")
x = ["Fold - 1", "Fold - 2", "Fold - 3", "Fold - 4", "Fold - 5"]

default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
plt.legend(loc="upper left")
plt.style.use("ggplot")
plt.show