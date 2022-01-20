from typing import Sized
import cv2
import numpy as np
from keras.models import load_model
model=load_model("/home/gianluca/Downloads/face-mask-detector-project/mask_detector.model") #loading the mask detection model

labels_dict={0:'With Mask',1:'Without Mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

size = 4
cam = cv2.VideoCapture(0) #Use camera 0

#loading the face detection model
classifier = cv2.CascadeClassifier('/home/gianluca/Downloads/face-mask-detector-project/haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cam.read()
    im=cv2.flip(im,1,1) 

    
    sized = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    
    faces = classifier.detectMultiScale(sized)

    
    for f in faces:
        (x, y, w, h) = [v * size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    
    #creating an exit key for the program
    if key == 27: #exit key = esc
        break

cam.release()


cv2.destroyAllWindows()