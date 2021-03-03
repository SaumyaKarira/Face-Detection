import cv2
import numpy as np
import face_recognition
import os

path = 'Images'
images = [] #all images imported from Images folder put here
classNames = []
myList = os.listdir(path)
print(myList)

for img in myList:
    currImage = cv2.imread(f'{path}/{img}')
    images.append(currImage)
    classNames.append(os.path.splitext(img)[0])
print(classNames)

imgElon = face_recognition.load_image_file('Images/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB) #Convert image to rgb

imgTest = face_recognition.load_image_file('Images/elon musk 2.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) #Convert image to rgb