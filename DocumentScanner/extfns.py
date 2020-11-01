import cv2 as cv
import numpy as np

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def resize(image, width=None, height=None,inter=cv.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)

    else:
        r = width / float(w)
        dim = (width, int(h*r))

    resized = cv.resize(image, dim, interpolation=inter)

    return resized

def rotate(image, angle, center=None,scale=1.0):
    (h,w) = image.shape[:2]

    # computing the center
    if center is None:
        center = (w//2,h//2)
    
    # getting the rotation matrix, passing the center,
    # the angle to rotate by and the scaling factor
    M = cv.getRotationMatrix2D(center,angle,scale)

    # rotating the image, passing the image, 
    # the transformation matrix, and the dimensions
    rotated = cv.warpAffine(image, M, (w,h))

    return rotated

def rotate_bound(image, angle):
    (h,w) = image.shape[:2]
    (cX,cY) = (w/2,h/2)

    # grabbing the sine and cosine
    # i.e the rotation component of the matrix
    M = cv.getRotationMatrix2D((cX,cY), -angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    # computing the new bounding dimension of the image
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))

    # adjusting the rotation matrix
    M[0,2] += (nW / 2) - cX
    M[1,2] += (nH / 2) - cY

    return cv.warpAffine(image, M, (nW, nH))

def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)

    return edged