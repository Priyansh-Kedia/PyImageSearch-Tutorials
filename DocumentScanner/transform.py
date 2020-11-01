import numpy as np
import cv2 as cv    

def order_points(points):
    """
        Initialise a list of coordinates that will be ordered
        such that the first entry in the list is the top-left,
        second is the top-right, third is the bottom-right, and
        the fourth entry is bottom-left 
    """
    rect = np.zeros((4,2), dtype='float32')
    
    """
    The top-left point will have the smallest sum, where as the 
    bottom-right point will have the largest sum among all four
    """
    sum = points.sum(axis=1)
    rect[0] = points[np.argmin(sum)]
    rect[2] = points[np.argmax(sum)]

    """
    The top-right will have the smallest difference, and 
    the bottom-left will have the largest difference
    """
    difference = np.diff(points, axis=1)
    rect[1] = points[np.argmin(difference)]
    rect[3] = points[np.argmax(difference)]

    return rect

def four_point_transform(image, pts):
    """
    Obtain a consistent order of points and 
    unpack them individually
    """
    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    """
    Compute the width of the new image, which will be the
    maximum distance between bottom-right and bottom-left or
    between top-right and top-left
    """
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA),int(widthB))

    """
    Compute the height of new image, which will be the 
    maximum distance between top-right and bottom-right or 
    between the top-left and bottom-left
    """
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    """ 
   	Now that we have the dimensions of the new image, construct
	the set of destination points to obtain a "birds eye view",
	(i.e. top-down view) of the image, again specifying points
	in the top-left, top-right, bottom-right, and bottom-left
	order
    """

    dst = np.array([
        [0,0],
        [maxWidth - 1,0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    """
    getPerspectiveTransform takes source and destination array,
    converts the source to the destination  
    """
    M = cv.getPerspectiveTransform(rect, dst)

    # warpPerspective takes image, the input array, and the size
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped