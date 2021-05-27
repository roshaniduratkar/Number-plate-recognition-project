import  numpy as np
import pytesseract as tes
import cv2

tes.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#cascade=cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

faceCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

img = cv2.imread(r'C:\Users\HP\Documents\numberplaterecognition\download (2).jpg')

def rescaleframe(frame,scale=1.50):#resizing image
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,125,255,cv2.THRESH_BINARY)

faces = faceCascade.detectMultiScale(thresh,scaleFactor=1.1,minNeighbors = 4,minSize=(25,25))
print(faces)
for (x,y,w,h) in faces:
    a, b = (int(0.002* img.shape[0]), int(0.025 * img.shape[1]))
    print(a,b)
    cropped_plate = img[y+a:y+h-a, x+b-4:x+w-b]
    cv2.imshow("cro",cropped_plate)

    resize_image = rescaleframe(cropped_plate)
    cv2.imshow('resizeimage', resize_image)

    kernel=np.ones((1,1), np.uint8)
    plate = cv2.dilate(resize_image,kernel,iterations=1)
    plate = cv2.erode(plate, kernel, iterations=1)
    gry = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    (thresh, thr) = cv2.threshold(gry, 115, 200, cv2.THRESH_BINARY)

    read = tes.image_to_string(thr)

    read=''.join(e for e in read if(e.isalnum()))
    print("number is:", read)

    rect = cv2.rectangle(img, (x + w, y + h), (x, y), (50, 100, 600), thickness=1)
    plate = cv2.rectangle(img, (x, y - h), (x + w, y), (50, 100, 600), cv2.FILLED)
    cv2.putText(plate, read, (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 50), thickness=1)
    cv2.imshow("text", plate)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

