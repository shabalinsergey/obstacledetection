
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob


def GetUndistortedImage(image_path, DistortionMatrix):
  
  ret = DistortionMatrix[0]
  mtx = DistortionMatrix[1]
  dist = DistortionMatrix[2]
  rvecs = DistortionMatrix[3]
  tvecs = DistortionMatrix[4]
   


  distorted_image = cv2.imread(image_path)

  height, width = distorted_image.shape[:2]
      
  # Refine camera matrix
  # Returns optimal camera matrix and a rectangular region of interest
  optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, 
                                                              (width,height), 
                                                              1, 
                                                              (width,height))

    # Undistort the image
    
  undistorted_image_gray = cv2.undistort(distorted_image, mtx, dist, None, 
                                      optimal_camera_matrix)

  undistorted_image=cv2.cvtColor(undistorted_image_gray, cv2.COLOR_BGR2RGB)
  cv2.imshow('undistoted',undistorted_image)
  return undistorted_image

def SaveUndistortedImage(distroted_img, distorted_img_filename):
  
  # Create the output file name by removing the '.jpg' part
  size = len(distorted_img_filename)
  new_filename = distorted_img_filename[:size - 4]
  new_filename = new_filename + '_undistorted.jpg'
  # Save the undistorted image
  cv2.imwrite(new_filename, distroted_img)  
  
  return 0