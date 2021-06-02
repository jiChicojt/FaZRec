import os
import cv2
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train():
    id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "_").lower()

                if not label in label_ids:
                    label_ids[label] = id
                    id += 1

                id_ = label_ids[label]

                img = Image.open(path).convert("L")
                size = (550, 550)
                r_img = img.resize(size, Image.ANTIALIAS)
                img_array = np.array(r_img, "uint8")

                faces = face_cascade.detectMultiScale(img_array)
                for (x, y, w, h) in faces:
                    roi = img_array[y: y + h, x: x + w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")

    print("Datos cargados exitosamente\n")