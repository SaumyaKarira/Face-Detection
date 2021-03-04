import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

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

#Encoding each image
def computeEncodins(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('AttendanceFile.csv', 'r+') as f:
        myData = f.readlines()
        #print(myData)
        nameList = []
        for line in myData:
            entry = line.split(',')
            #name = entry[0]
            nameList.append(entry[0])
        print("list",nameList)
        if name not in nameList:
            print("name" ,name)
            now = datetime.now()
            time = now.strftime('%H:%M:%S') #Format
            f.writelines(f'\n{name},{time}')


encodeListForImages = computeEncodins(images)
print("Encoding Completed")

#Find matches between encodings, comparing with webcam
#Initialise with webcam

capture = cv2.VideoCapture(0) #0 is id
#To get each frame
while True:
    success, img = capture.read()
    #speed increases by reducing the image
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    #image, pixel size(not specifying), dst: None, Scale(0.25 is 1/4th of the scale)

    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    #There might be multiple faces in current frame. To send their locations to encode
    facesCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurrFrame = face_recognition.face_encodings(imgSmall,facesCurrFrame)

    # Finding the matches. Iterate through all the faces founded in current frame and then compare it with
    # all the face encodings we found before(downloaded images)
    for encodeFace, faceLoc in zip(encodeCurrFrame,facesCurrFrame): #zip used because we want them in same loop
        matches = face_recognition.compare_faces(encodeListForImages, encodeFace)
        faceDis = face_recognition.face_distance(encodeListForImages, encodeFace)
        print(faceDis)
        #To the lowest distance, as will be a better match
        matchIndex = np.nanargmin(faceDis)

        #Printing the name of person which matches in current frame with the stored one
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 #To convert to original size,for properly locating the rectangle
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_ITALIC,1,(255,255,255),2)
            markAttendance(name)

        cv2.imshow("Webcam Image", img)
        cv2.waitKey(1)



