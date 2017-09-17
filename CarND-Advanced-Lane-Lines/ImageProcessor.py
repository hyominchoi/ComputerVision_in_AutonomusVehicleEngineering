import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from numpy.linalg import inv


# Calibrate Camera
def calibrateCamera(image_path ='camera_cal/calibration*.jpg',
                     nx=9, ny=6, image_shape=(720, 1280)):
    """
    this function calibrates camera distortion parameters
    by reading the chessboard images in the image_path
        
    :param image_path : chessboards image path
    :param nx : number of grids in x-driection
    :param ny : number of grids in y-direction
    :param image_shape : shape of input images  
    
    :returns: ret, mtx, dist, rvecs, tvecs
    """

    # Read in an image
    images = glob.glob(image_path)
    objpoints = []
    imgpoints = []

    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) #z-coordinate is all zero

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)   

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    return ret, mtx, dist, rvecs, tvecs

# Undistort Images using Camera Calibration results
def undistortImage(img, ret, mtx, dist):
    """ 
    undistort image

    :param img : distorted image
    :param ret, mtx, dist : results from calibrateCamera()

    :returns: corrected, distortion-free image
    """
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


# Gradient & Color Threshold 
def imageThresholding(img, s_thresh=(100, 255), h_thresh=(15, 100),
                      sx_thresh=(20, 100), sy_thresh=(100, 100)):
    """
    This function takes a distortion-free image and applies
    color channel conversion and Sobel-x, Sobel-y gradient.
    Once these are applied, we take pixels within the specified
    threshold ranges, and convert the original image into
    binary image.

    :param img : distortion-free image
    :param s_thresh : HSL color channel S-channel threshold
    :param h_thresh : HSL color channel H-channel threshold
    :param sx_thresh: gradient value (Sobel-x) threshold
    :param sy_thresh: gradient value (Sobel-y) threshold

    :returns: combined_binary, pixels selected after thresholding

    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,2]
    l_channel = hsv[:,:,1]
    
    # Sobel x
    # Take the derivative in x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) 
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Sobel y #
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
    # Absolute y derivative to accentuate lines away from vertical
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255*abs_sobely/np.max((abs_sobely)))
    
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[(s_binary == 1) | (sxbinary == 1) | (sybinary == 1)] = 1
    combined_binary[((s_binary == 1) & (h_binary == 1))| (sxbinary == 1) | (sybinary == 1) ] = 1
    return combined_binary

# Persepective Transform
def transformMatrix():
    """
    Find conversion between source region (src) and destination region(dst).

    :returns: 
        M : conversion matrix, numpy.array
        Minv : inverse matrix, numpy.array
    """

    src = np.float32([[180, 700], [490, 500], [790, 500], [1100,700]])
    dst = np.float32([[180, 700], [180, 450], [1100, 450], [1100, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = inv(M)
    return M, Minv

def warper(img, M):
    """
    Converts img to bird-eye view image.

    :param img : original image, numpy array
    :param M : conversion matrix, numpy array

    :returns: warped image
    """
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped
