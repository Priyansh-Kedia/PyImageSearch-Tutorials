from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
from extfns import *
import argparse
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
ratio = image.shape[0] / 500.0
original = image.copy()
image = resize(image, height = 500)

"""
Convert the image to grayscale, blur it, and find edges
in the image
"""
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5,5),0)
edged = cv.Canny(gray, 75, 200)
# edged = auto_canny(gray)

# Show the original and the edge detected image
cv.imshow("Image", image)
cv.imshow("Edged",edged)
cv.waitKey(0)
cv.destroyAllWindows()  

"""
Find the contours in the edged image, keeping only the
largest ones, and initialise the screen contour
"""
contours = cv.findContours(edged.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
contours = grab_contours(contours)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

# Loop over the contours
for c in contours:
    perimeter = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02*perimeter, True)
    # If the polygon has 4 sides, we have our document
    if len(approx) == 4:
        screenCount = approx
        break

cv.drawContours(image, [screenCount], -1, (0, 255, 0), 2)
cv.imshow("Outline", image)
cv.waitKey(0)
cv.destroyAllWindows()

"""
Applying the bird-eye view to the scanned document
"""
warped = four_point_transform(original, screenCount.reshape(4,2) * ratio)

"""
Applying grayscale to the warped image, then threshold it
"""
warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
# T = threshold_local(warped, 21, offset=3, method='gaussian')
T = cv.adaptiveThreshold(warped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,5 ,5)
warped = (warped > T).astype('uint8') * 255

cv.imshow("original", resize(original, height=650))
cv.imshow("warped", resize(warped, height=650))
cv.waitKey(0)