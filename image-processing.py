import cv2
import time
import numpy as np
from numpy.core.numeric import outer

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('Output.avi',fourcc,20.0,(640,480))
capture = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = capture.read()
bg = np.flip(bg,axis=1)

while(capture.isOpened()):
    ret, image = capture.read()
    if not ret:
        break

    image = np.flip(image,axis=1)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    lowerRed = np.array([0,120,50])
    upperRed = np.array([180,255,255])
    mask1 = cv2.inRange(hsv,lowerRed,upperRed)

    lowerRed = np.array([170,120,70])
    upperRed = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lowerRed,upperRed)

    mask1 = mask1 + mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(mask1)

    red1 = cv2.bitwise_and(image,image,mask=mask2)
    red2 = cv2.bitwise_and(bg,bg,mask=mask1)

    finalOutput = cv2.addWeighted(red1,1,red2,1,0)
    output_file.write(finalOutput)
    cv2.imshow('magic',finalOutput)
    cv2.waitKey(1)

capture.release()
outer.release()
cv2.destroyAllWindows()




