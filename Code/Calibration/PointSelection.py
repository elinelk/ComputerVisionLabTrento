import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
## CAPTURE VIDEO FROM CAMERA
cap = cv2.VideoCapture('/Users/elinelillebokarlsen/ComputerVisionLabTrento/Videos/out9.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
print(f"Size of frame width:{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}") 
i = 0
j =0
while True:
# Capture frame-by-frame
     ret, frame = cap.read()
     cam1 = frame[:,0:897]
     cam2 = frame[:,897:2049]
     cam3 = frame[:,2049:3201]
     cam4 = frame[:,3201:4096]
     gray = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # # Our operations on the frame come here
    
     # Find the chess board corner
     if i == 20:
         plt.imshow(cam2)
         plt.title('Click on points')
         plt.axis('on')
         #cv2.imwrite("/Users/elinelillebokarlsen/Downloads/inspectionpic.png", cam2)

         # Let user click on points and record the coordinates
         points = plt.ginput(n=-1, timeout=0)  # `n` is the number of points, `timeout` is how long to wait. Here, wait indefinitely.
         plt.close()
     elif i > 20:
         break
     i += 1
     cv2.imshow('Camera2', cam2)
     while True:
         if cv2.waitKey(10):
             break
     if cv2.waitKey(1) == ord('q'):
         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

