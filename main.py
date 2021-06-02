import cv2
import pickle
import os
from src.trainer import train

face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

print('Espere un momento mientras se cargan los datos...')
train()
recognizer.read("trainer.yml")

opt = 0

while opt == 0:
    decision = input('¿Qué operación desea realizar?\n1. Agregar a la base de datos\n2. Buscar coincidencias\n3. Salir\n')
    opt = int(decision)

    if opt == 1:
        name = input('Ingrese el nombre de la persona a escanear> ')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        folder_dir = os.path.join(BASE_DIR, "src", "images", name)
        os.makedirs(folder_dir, exist_ok=True)

        cam = cv2.VideoCapture(0)
        sample = 0

        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                sample += 1
                print(sample)
                roig = gray[y: y + h, x: x + w]

                cv2.imwrite(os.path.join(folder_dir, str(sample) + ".jpg"), roig)

                color = (204, 0, 204)
                stroke = 2
                xf = x + w
                yf = y + h
                cv2.rectangle(frame, (x, y), (xf, yf), color, stroke)
                cv2.waitKey(100)

            cv2.imshow('FaZRec - Scanner', frame)
            cv2.waitKey(1)
            if sample > 99:
                break

        cam.release()
        cv2.destroyAllWindows()

        train()

        opt = 0
    elif opt == 2:
        labels = {"name": 1}
        with open("labels.pickle", "rb") as f:
            olabels = pickle.load(f)
            labels = {v: k.replace("_", " ").upper() for k, v in olabels.items()}

        cap = cv2.VideoCapture(0)

        while (True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roig = gray[y: y + h, x: x + w]

                id_, conf = recognizer.predict(roig)
                if conf >= 45:
                    font = cv2.FONT_HERSHEY_DUPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

                img_item = "me.png"
                cv2.imwrite(img_item, roig)

                color = (204, 0, 204)
                stroke = 2
                xf = x + w
                yf = y + h
                cv2.rectangle(frame, (x, y), (xf, yf), color, stroke)

            cv2.imshow('FaZRec - Recognition', frame)
            if cv2.waitKey(20) & 0xFF == ord('x'):
                break

        cap.read()
        cv2.destroyAllWindows()
        opt = 0
    elif opt == 3:
        print('Gracias por utilizar FaZrec')
        opt = 1
    else:
        print('La operación no está definida')
        opt = 0
        pass