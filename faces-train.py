import os
import cv2
import numpy as np
from PIL import Image
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recoginizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_label = []
x_train = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
   
 
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1

            id_ = label_ids[label]
           

            pil_image = Image.open(path).convert("L") # grayscale
            image_array = np.array(pil_image) 
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
      
            faces = face_cascade.detectMultiScale(image_array, 1.5, 5)
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id_)



with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids,f)

recoginizer.train(x_train, np.array(y_label))
recoginizer.save("trainner.yml")