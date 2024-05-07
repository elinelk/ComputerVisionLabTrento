import numpy as np
import cv2
import functions

def bg_update(current_frame, prev_bg, alpha):
    bg = alpha * current_frame + (1 - alpha) * prev_bg

    # print(f"Background dtype (before): {bg.dtype}")
    # print(f"Background dtype (after): {bg.dtype}")

    bg = np.uint8(bg)
    
    return bg

background = None
MAX_FRAMES = 1000
THRESH = 50
MAXVAL = 255
ALPHA = 0.04
MAX_FRAMES = 1000
LEARNING_RATE = -1  # alpha
HISTORY = 200       # t
N_MIXTURES = 5    # K (number of gaussians)
BACKGROUND_RATIO = 0.1 # Gaussian threshold
NOISE_SIGMA = 1     
MOG_VERSION = 2
## CAPTURE VIDEO FROM CAMERA

cap = cv2.VideoCapture('/Users/elinelillebokarlsen/ComputerVision/out9.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

if MOG_VERSION == 1:
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA)
elif MOG_VERSION == 2:
    fgbg = cv2.createBackgroundSubtractorMOG2()
else:
    print(f"Unknown MOG version {MOG_VERSION}")
    quit(0)


for t in range(MAX_FRAMES):
# Capture frame-by-frame
     ret, frame = cap.read()
     cam1 = frame[:,0:897]
     cam2 = frame[:,897:2049]
     cam3 = frame[:,2049:3201]
     cam4 = frame[:,3201:4096]
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # # Our operations on the frame come here

     #fgmask1 = fgbg.apply(cam2, LEARNING_RATE)
     #fgmask2 = fgbg.apply(cam3, LEARNING_RATE)
     #if MOG_VERSION == 2:
     #    bg = fgbg.getBackgroundImage()
     #    graybg = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
     #    orb = cv2.ORB.create()
     #    dst = cv2.cornerHarris(graybg,2,3,0.04)
     #    feat,desc = orb.detectAndCompute(bg, None)
     #    dst = cv2.dilate(dst,None)
     #    bg[dst>0.01*dst.max()]=[0,0,255]
     #    cv2.imshow('bg', bg)
 
     #cv2.imshow('fgmask', fgmask)
     #cv2.imshow('frame', frame)
 
     # Wait and exit if q is pressed
     if cv2.waitKey(10) == ord('q') or not ret:
         break
     frame_gray1 = cv2.cvtColor(cam2, cv2.COLOR_RGB2GRAY)
     frame_gray2 = cv2.cvtColor(cam3, cv2.COLOR_RGB2GRAY)
     if t == 0:
        # Train background with first frame
        background1 = frame_gray1
        background2 = frame_gray2
     else:
        # Background subtraction
        diff1 = cv2.absdiff(background1, frame_gray1)
        diff2 = cv2.absdiff(background2, frame_gray2)

        # Mask thresholding
        ret1, motion_mask1 = cv2.threshold(diff1, THRESH, MAXVAL, cv2.THRESH_BINARY)
        ret2, motion_mask2 = cv2.threshold(diff2, THRESH, MAXVAL, cv2.THRESH_BINARY)

        # Update background
        background1 = bg_update(frame_gray1, background1, alpha = ALPHA)
        background2 = bg_update(frame_gray2, background2, alpha = ALPHA)
        dst1 = cv2.cornerHarris(background1,2,3,0.04)
        dst2 = cv2.cornerHarris(background2,2,3,0.04)
        dst1 = cv2.dilate(dst1,None)
        dst2 = cv2.dilate(dst2,None)
        cam2[dst1>0.01*dst1.max()]=[0,0,255]
        cam3[dst2>0.01*dst2.max()]=[0,0,255]

        # Detect Harris corners and compute descriptors (ORB in this example)
        orb = cv2.BRISK.create()
        kp1, des1 = orb.detectAndCompute(background1, None)
        kp2, des2 = orb.detectAndCompute(background2, None)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)  

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros_like(points1)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography
        height, width = background1.shape
        #width += 400
        im2_aligned = cv2.warpPerspective(background2, H, (width, height))

        # Create a panorama
        panorama = cv2.addWeighted(background1, 0.5, im2_aligned, 0.5, 0)

        # Display the resulting frames
        #cv2.imshow('Frame', frame)
        #cv2.imshow('Motion mask', motion_mask)
        cv2.imshow('Corners1', cam2)
        cv2.imshow('Corners2', cam3)
        cv2.imshow('warped', panorama)
        # Wait and exit if q is pressed
        if cv2.waitKey(1) == ord('q') or not ret:
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()