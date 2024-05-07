import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
## CAPTURE VIDEO FROM CAMERA

object_points = np.array([
    [0, 0, 0],
    [0.914, 4.26, 0],
    [6.748, 1.22, 0],
    [5.79, 5.79, 0],
    [7.62, 12.495, 0],
    [3.12, 8.325, 0],
    [3.12, 14.325, 0],
    [2.21,17.325,0],
    [0.914, 0,0],
    [7.62,0,0]
])

image_points = np.array([
    [144,1594],
    [231,1346],
    [860,1587],
    [786,1059],
    [953,381],
    [455,1146],
    [440,455],
    [331,211],
    [230,1621],
    [998,1696]
])

object_points = np.array([object_points], dtype=np.float32)
image_points = np.array([image_points], dtype=np.float32)



cap = cv2.VideoCapture('/Users/elinelillebokarlsen/ComputerVisionLabTrento/Videos/out9.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
height, width = frame.shape[:2]
success, camera_matrix, distortion_coefficients, rotation_vector, translation_vector = cv2.calibrateCamera(object_points, image_points, (width,height), None, None)
new_camera_matrix, region_of_interest = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, (width, height), 1, (width, height)
    )
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
     undistorted_image = cv2.undistort(cam2, camera_matrix, distortion_coefficients, None, new_camera_matrix)
     cv2.imshow('Camera2', cam2)
     cv2.imshow('Undistorted', undistorted_image)
     while True:
         if cv2.waitKey(10):
             break
     if cv2.waitKey(1) == ord('q'):
         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

