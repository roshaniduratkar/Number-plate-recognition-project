import cv2 as cv
#from  matplotlib import pyplot as plt
import  numpy as np
import pytesseract as tes
import imutils

tes.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cascade=cv.CascadeClassifier("haarcascade_russian_plate_number.xml")

def rescaleframe(frame,scale=1.5):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

image=cv.imread(r'C:\Users\HP\Documents\numberplaterecognition\download.jpg')

gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

ret,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)

nplate=cascade.detectMultiScale(thresh,1.1,4)
for (x,y,w,h)in nplate:
    a,b=(int(0.002*image.shape[0]),int(0.025*image.shape[1]))
    cropped_plate=image[y+a:y+h-a, x+b:x+w-b]
    cv.imshow("crop", cropped_plate)
    resize_image = rescaleframe(cropped_plate)
    cv.imshow('resizeimage', resize_image)

    grplate = cv.cvtColor(resize_image, cv.COLOR_BGR2GRAY)
    (thresh, plate) = cv.threshold(grplate, 127, 200, cv.THRESH_BINARY)
    #cv.imshow("newthresh",plate)


    read = tes.image_to_string(plate)
    read = ''.join(e for e in read if e.isalnum())
    cv.rectangle(image, (x+w, y+h), (x, y), (100, 50, 600), thickness=2)
    cv.rectangle(image, (x , y-h), (x+w, y), (100, 50, 600),cv.FILLED)
    cv.putText(image, read, (x+5, y-10), cv.FONT_HERSHEY_PLAIN, 1, (100, 250, 50), thickness=2)
    print("number is:",read)
cv.imshow("re",image)


cv.waitKey(0)
cv.destroyAllWindows()