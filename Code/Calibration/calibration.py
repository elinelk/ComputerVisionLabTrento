import numpy as np
import cv2
import json
import functions

def average_corner_distance(corners1, corners2):
    # Calculate the Euclidean distance between corresponding corners
    # and take the average
    distances = np.linalg.norm(corners1 - corners2, axis=1)
    return np.mean(distances)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 

# Define the dimensions of the chessboard
chessboard_size = (9, 6)  # Number of inner corners per a chessboard row and column

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
threshold = 35  # Define the threshold distance


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
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # # Our operations on the frame come here
     # Find the chess board corner
     if i > 750 and i < 1250:
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
     # If found, add object points, image points (after refining them)
        if ret == True:
           corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
           if objpoints:  # Check only if there are already corners in the list
            mean_corners = np.mean(objpoints, axis=0)
            distance = average_corner_distance(corners2, mean_corners)
            if distance > threshold:
                objpoints.append(corners)
                imgpoints.append(frame)
                print(f"{j}: True")
                j+=1
           else:
            objpoints.append(corners)
            imgpoints.append(frame)
            print(f"{j}: True")
            j+=1
            # Draw and display the corners
           cv2.drawChessboardCorners(cam2, chessboard_size, corners, ret)
           cv2.imshow('img', cam2)
        cv2.imshow('img', cam2)
     elif i >1250:
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
