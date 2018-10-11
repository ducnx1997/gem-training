import cv2
import numpy as np
 
params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 200

params.filterByArea = True
params.minArea = 1500

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.87

params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

img = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)

keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
img_with_keypoints = cv2.drawKeypoints(
    img, 
    keypoints, 
    np.array([]),   
    (0,0,255), 
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints", img_with_keypoints)
cv2.waitKey(0)