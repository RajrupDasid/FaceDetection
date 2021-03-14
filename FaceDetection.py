import cv2 
import numpy as np
face_cascade = cv2.CascadeClassifier('C:/Users/Rajrup Das/source/repos/FaceDetection/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        img_item = "My-image.png"
        cv2.imwrite(img_item, roi_gray)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()