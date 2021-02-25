import cv2
import os

def imageResizing(im):
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces = classifier.detectMultiScale(im, # stream 
                                        scaleFactor=1.10, # change these parameters to improve your video processing performance
                                        minNeighbors=20, 
                                        minSize=(30, 30) # min image detection size
                                        ) 
    for (x, y, w, h) in faces:
            # saving faces according to detected coordinates 
            sub_face = im[y:y+h, x:x+w]
            sub_face = cv2.resize(sub_face,(256,256))
            
    return sub_face