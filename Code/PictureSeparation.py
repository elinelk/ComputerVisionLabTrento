import numpy as np
import cv2
import functions

## CAPTURE VIDEO FROM CAMERA

cap = cv2.VideoCapture('/Users/elinelillebokarlsen/ComputerVision/out9.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print(f"Size of frame width:{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
while True:
# Capture frame-by-frame
     ret, frame = cap.read()
     cam1 = frame[:,0:897]
     cam2 = frame[:,897:2049]
     cam3 = frame[:,2049:3201]
     cam4 = frame[:,3201:4096]
     #cv2.rectangle(frame, (0, 0), (896, 1792), (0, 255, 0), 1)

     #cv2.rectangle(frame, (896, 0), (2048, 1792), (0, 0, 255), 1)
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # # Our operations on the frame come here
     stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)
     grey2 = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)
     grey3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2GRAY)
     disparity = stereo.compute(grey2, grey3)
     # Display the resulting frame

     #cv2.imshow('Camera1', cam1)
     cv2.imshow('Camera2', cam2)
     cv2.imshow('Camera3', cam3)
     #cv2.imshow('Camera4', cam4)
     cv2.imshow("Stiched image",disparity)
     while True:
         if cv2.waitKey(10):
             break
     if cv2.waitKey(1) == ord('q'):
         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()