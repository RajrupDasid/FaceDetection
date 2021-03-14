import cv2 
import numpy as np
import pickle
face_cascade = cv2.CascadeClassifier('C:/Users/Rajrup Das/source/repos/FaceDetection/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained.yml")


labels ={"person_name":1}
with open ("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k  for k,v in og_labels.items()}



cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #recognize deep learning model
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf<=85:
            print(id_)
            print(labels[id_])
        img_item = "My-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255,0,0)
        stroke = 2
        
        end_cord_x = x+y
        end_cord_y = y+h

        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)




    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()