import argparse
import cv2 as cv
import extfns as ef

# Construct an argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="path to input image")
args = vars(ap.parse_args())

# loading the image
image = cv.imread(args["image"])
cv.imshow("image",image)
cv.waitKey(0)

# converting the image to grayscale
grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("gray",grayScale)
cv.waitKey(0)

# edge detection
"""
This is the canny edge detection method that we use
The second param is the minimum threshold value 
The third param is the maximum threshold value
"""
edged = cv.Canny(grayScale, 30,150)
cv.imshow("edged",edged)
cv.waitKey(0)

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 0
# (black; background), thereby segmenting the image
thresh = cv.threshold(grayScale, 225, 255, cv.THRESH_BINARY_INV)[1]
cv.imshow("Thresh", thresh)
cv.waitKey(0)

"""
Now we will find the contours int the image so as to 
highlight the different blocks. This would be done on 
the threshold image
"""
contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = ef.grab_contours(contours)
output = image.copy()

"""
The third param is negative, which means 
that all the contours will be drawn
The fourth param is the color, and the last 
param is the thickness
"""

text = "I found {} objects!".format(len(contours))
cv.putText(output, text, (10, 25),  cv.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)

for c in contours:
    cv.drawContours(output,[c],-1,(240,0,150),3)

cv.imshow("contours", output)
cv.waitKey(0)


# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv.erode(mask, None, iterations=5)
cv.imshow("Eroded", mask)
cv.waitKey(0)

# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv.dilate(mask, None, iterations=5)
cv.imshow("Dilated", mask)
cv.waitKey(0)

"""
A typical operation we may want to apply is to take our mask and
apply a bitwise AND to our input image, keeping only the masked
regions
"""
mask = thresh.copy()
output = cv.bitwise_and(image, image, mask=mask)
cv.imshow("Output", output)
cv.waitKey(0)