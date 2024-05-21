import cv2
import numpy as np


#Using blending window 40
image1 = cv2.imread('/Users/marihetlesaeter/Desktop/stitch_1.jpg')
image2 = cv2.imread('/Users/marihetlesaeter/Desktop/stitch_2.jpg')


def showMatches(matchesImage):
    cv2.imshow('Top Matches', matchesImage)
    cv2.imwrite('topMatches.jpg', matchesImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def createWindow(image1, image2):
    heightStitched = image1.shape[0]
    widthStitched = image1.shape[1] + image2.shape[1]
    return heightStitched, widthStitched


def getHomography(image1,image2):
    ratio = 0.70 #The ratio is usually between 0.6 and 0.8

    #Converting to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    #Find the keypoints and descriptor using SIFT
    sift = cv2.SIFT.create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    #Getting the 2 best matches for each descriptor
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2) 

    #Applying the ratio test
    goodMatches=[]
    goodPoints = []
    for m in matches:
        m1,m2 = m[0], m[1]
        if m1.distance < ratio * m2.distance: 
            goodMatches.append([m1])
            goodPoints.append((m1.trainIdx, m1.queryIdx))


    #Drawing the top matches
    #matchesImage = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, goodMatches, None, flags=2)
    #showMatches(matchesImage)
    
    #Finding the coordinates of the good matches and computing the homography matrix
    if len(goodPoints) >= 4:
        image1Keypoints = np.float32([keypoints1[m1.queryIdx].pt for [m1] in goodMatches])
        image2Keypoints = np.float32([keypoints2[m1.trainIdx].pt for [m1] in goodMatches])
        H, _ = cv2.findHomography(image2Keypoints, image1Keypoints, cv2.RANSAC,5.0)
    else: print("Not enough matches to compute homography")
    return H



def createMask(image1,image2,image):
    #Size of blending region
    blendingRegionSize= int(40)

    #Finding the barrier (to the left of where the first image ends)
    barrier = image1.shape[1] - int(blendingRegionSize*4)

    #Dimension calculation for the new, stitched image and fill with zeros
    heightStitched, widthStitched = createWindow(image1, image2)
    mask = np.zeros((heightStitched, widthStitched))


    #Setting the area to the left of the left barrier, and to the right of the right barrier to the original pixel value
    if image == 'leftImage':
        mask[:, barrier- blendingRegionSize:barrier+ blendingRegionSize] = np.tile(np.linspace(1, 0, 2 * blendingRegionSize ).T, (heightStitched, 1)) #Defining pixel value of the blending area
        mask[:, :barrier-blendingRegionSize] = 1 #Setting all pixels to the left of the barrier to 1
    else:
        mask[:, barrier- blendingRegionSize:barrier +  blendingRegionSize] = np.tile(np.linspace(0, 1, 2 *  blendingRegionSize ).T, (heightStitched, 1))
        mask[:, barrier+blendingRegionSize:] = 1 #Setting all pixels to the right of the barrier to 1

    return cv2.merge([mask, mask, mask]) #merging into a color image


def stitchingImages(image1,image2):
    #Creating the final image with all pixel values set to zero
    heightStitched, widthStitched = createWindow(image1, image2)
    emptyImage = np.zeros((heightStitched, widthStitched, 3)) #3 represents rgb

    #Dimensions for the image to the left
    leftHeight = image1.shape[0]
    leftWidth = image1.shape[1]

    #Calculate homography matrix, and perform perspective transformation to make the right part adjust to the left part of the image
    H = getHomography(image1,image2)
    perspectiveTransformation = cv2.warpPerspective(image2, H, (widthStitched, heightStitched))

    #Calculating the pixel values for the right part of the final result
    rightMask = createMask(image1,image2,image='rightImage')
    rightPart = perspectiveTransformation*rightMask

    #Calculating the pixel values for the left part of the final result
    leftMask = createMask(image1,image2,image='leftImage')
    emptyImage[0:leftHeight, 0:leftWidth, :] = image1
    emptyImage *= leftMask #Multiplying each pixel in the left part with the pixel values of the left mask.Pixels inside blending area will blend gradually
    leftPart = emptyImage

    #Combining the left and right part into the final stitched image
    finalResult=leftPart + rightPart
    return finalResult


#Saving the final result as a image
stitchedImage = stitchingImages(image1,image2)
cv2.imwrite('result.jpg', stitchedImage)






    
