import numpy as np
import cv2
import json
import functions

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

## CAPTURE VIDEO FROM CAMERA

cap = cv2.VideoCapture('/Users/elinelillebokarlsen/ComputerVisionLabTrento/Videos/out9safe.mp4')
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
     #cv2.rectangle(frame, (0, 0), (896, 1792), (0, 255, 0), 1)

     #cv2.rectangle(frame, (896, 0), (2048, 1792), (0, 0, 255), 1)
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # # Our operations on the frame come here
     # Find the chess board corner
     if i > 750 and i < 1000:
        ret, corners = cv2.findChessboardCorners(gray, (6,6), None)
     # If found, add object points, image points (after refining them)
        if ret == True:
           j+=1
           print(f"{j}: True")
           objpoints.append(objp)
           corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
           imgpoints.append(corners2)
           cv2.drawChessboardCorners(gray, (6,6), corners2, ret)
        
        cv2.imshow('img', gray)
     elif i >1000:
        break
    # Draw and display the corners
     i +=1
     #cv2.imshow('Camera2', cam2)
     while True:
         if cv2.waitKey(10):
             break
     if cv2.waitKey(1) == ord('q'):
         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Save the calibration results to a file
calibration_data = {
    'camera_matrix': mtx.tolist(),
    'distortion_coefficients': dist.tolist(),
    'rotation_vectors': [np.array(vec).tolist() for vec in rvecs],
    'translation_vectors': [np.array(vec).tolist() for vec in tvecs]
}

with open('/Users/elinelillebokarlsen/ComputerVisionLabTrento/Output/calibration_data.json', 'w') as f:
    json.dump(calibration_data, f, indent=4)


print("Calibration data saved to calibration_data.json")
