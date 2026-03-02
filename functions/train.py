import cv2
import os
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR.parent / "userInfo" / "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()



def trainImg():
    if len(os.listdir(dataset_path)) == 0:
        return "Not enough data to train!"
    
    else:
        faces = []
        labels = []
        label_map = {}
        current_label = 0
        
        for person in os.listdir(dataset_path):
            label_map[current_label] = person
            person_path = os.path.join(dataset_path, person)

            for image_name in os.listdir(person_path):
                img_path = os.path.join(person_path, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(current_label)

            current_label += 1

        recognizer.train(faces, np.array(labels))
        recognizer.save("trainer/trainer.yml")

        np.save("trainer/labels.npy", label_map)

        return "Training Complete"

