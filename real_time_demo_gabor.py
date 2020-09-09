import numpy as np
import cv2
import pickle 
from joblib import dump, load

classifer = load('ck_gabor_normalized_float32_1.joblib')
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 2):
        kern = cv2.getGaborKernel((ksize, ksize), 8.0, theta, 25.0, 15.0, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

filters = build_filters()
# getting single frame and identifying the face
def start():
    cap = cv2.VideoCapture(0)
    while True:
        #try:
            ret, img = cap.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            emotion='unkown'
            for(x,y,w,h) in faces:
                res = process(cv2.resize(gray[y:y+h,x:x+w],(64,64)), filters)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                localarray = np.matrix(np.array(res.ravel(),dtype=np.float32))#converting 2d points array into 1d array
                localarray = localarray / 255
                cv2.putText(img,str(classifer.predict(localarray)),(x,y+h), font, 1,(255,255,255),2)
        #except:
            #continue
            cv2.imshow('emotion',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
start()

