import cv2 as cv
import extfns as ef

image = cv.imread("jp.jpg")

"""
height is the number of rows
width is the number of columns
depth is the number of channels
"""
(h,w,d) = image.shape

# accessing the BGR pixel at 100, 50
(B, G, R) = image[100, 50]

"""
extract a 100x100 pixel square ROI (Region of Interest) from the
input image starting at x=255,y=30 at ending at x=355,y=130
"""
roi = image[30:130, 255:355]


# resizing the image ignoring the aspect ratio
resized = cv.resize(image, (200,200))
# cv.imshow("Resized image", resized)

"""
Let us fix the width to 300px and calculate the height 
with respect to the aspect ratio
"""
r = 300.0 / w
dim = (300, int(h*r))
newResized = ef.resize(image, width=300)

"""
Rotating the image, we will rotate it by 45 degrees
We will first compute the image center, then construct 
the rotation matrix, see the ext function for the same
"""
rotated = ef.rotate(image, angle=-45)

"""
Rotating the image within the bounds so that the image
does not cut on the edges
"""

rotatedNew = ef.rotate_bound(image, 45)

"""
we blur an image to reduce the noise
We will use Gaussian Blur 
We use 11x11 kernel. Larger kernels yield a more
blurry image, smaller will yield a lesser blurry image
"""
blurred = cv.GaussianBlur(image, (11,11), 0)

"""
We make a rectangle on the image
We will duplicate the image first
The second and third param are the ends of rectangle
The fourth param is the color, in BGR format
"""
output = image.copy()
cv.rectangle(output, (255, 30),(355,130), (0,0,255),2)
cv.imshow("rectangle", output)
cv.waitKey(0)
# We can put other shapes similarly

"""
Now we will put text on the image
The third param is the starting point for the text
The fourth param is the font, you can find more fonts here https://docs.opencv.org/3.4.1/d0/de1/group__core.html#ga0f9314ea6e35f99bb23f29567fc16e11
Fourth param is the scale, which is the font size multiplier
The last param is the thickness of the stroke
"""
output = image.copy()
cv.putText(output, "The first tutorial", (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
cv.imshow("With text", output)
cv.waitKey(0)