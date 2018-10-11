import cv2
import numpy as np

img = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)

detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
img_with_keypoints = cv2.drawKeypoints(
    img, 
    keypoints, 
    np.array([]),   
    (0,255,0), 
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints", img_with_keypoints)
cv2.waitKey(0)