import numpy as np
import cv2
import json
import functions

def load_calibration_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    camera_matrix = np.array(data['camera_matrix'])
    distortion_coefficients = np.array(data['distortion_coefficients'])
    return camera_matrix, distortion_coefficients


camera_matrix, dist_coeffs = load_calibration_data('/Users/elinelillebokarlsen/ComputerVisionLabTrento/Output/calibration_data.json')

undistortVideo = cv2.VideoCapture('/Users/elinelillebokarlsen/ComputerVisionLabTrento/Videos/out9.mp4')
if not undistortVideo.isOpened():
    print("Cannot open camera")
    exit()

while True:
# Capture frame-by-frame
     ret, frame = undistortVideo.read()
     unVid = frame[:,897:2049]
     h, w = unVid.shape[:2]
     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
     
     #cv2.rectangle(frame, (0, 0), (896, 1792), (0, 255, 0), 1)

     #cv2.rectangle(frame, (896, 0), (2048, 1792), (0, 0, 255), 1)
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # # Our operations on the frame come here
     # undistort
     undistorted_img = cv2.undistort(unVid, camera_matrix, dist_coeffs, None, newcameramtx)
     # Display the resulting frame
     # Crop the image
     x, y, w, h = roi
     undistorted_img = undistorted_img[y:y+h, x:x+w]
    # Draw and display the corners

     cv2.imshow('Undistorted image', undistorted_img)
     cv2.imshow("original", unVid)
     #cv2.imshow('Camera2', cam2)
     while True:
         if cv2.waitKey(10):
             break
     if cv2.waitKey(1) == ord('q'):
         break
     
undistortVideo.release()
cv2.destroyAllWindows()
