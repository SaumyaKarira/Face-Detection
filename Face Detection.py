import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('Images/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB) #Convert image to rgb

imgTest = face_recognition.load_image_file('Images/elon musk 2.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) #Convert image to rgb

faceLoc = face_recognition.face_locations(imgElon)[0] #Locating face
#face location gives 4 values i.e corners of a rectangle
print(faceLoc)
encodeElon = face_recognition.face_encodings(imgElon)[0] #encode face
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)#to detect where face is
# image,corners of image, colour of rectangle, thickness

faceLocTest = face_recognition.face_locations(imgTest)[0] #Locating face in Test image
#face location gives 4 values i.e corners of a rectangle
print(faceLocTest)
encodeElonTest = face_recognition.face_encodings(imgTest)[0] #encode face
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)#to detect where face is
# image,corners of image, colour of rectangle, thickness

#Comparing both images(through encoding);linear svm(support vector machine) used to compare and finding the distance
result = face_recognition.compare_faces([encodeElon],encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon],encodeElonTest) #lower the distance, better the match
print(result,faceDis)
cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50), cv2.FONT_ITALIC,1, (0,0,255), 2)
# image, result(true/false), face distance(round off to 2 decimal places), origin, font, scale, colour, thickness

cv2.imshow("Elon Musk",imgElon)
cv2.imshow("Elon Musk Test",imgTest)
cv2.waitKey(0)