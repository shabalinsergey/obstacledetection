#Author: Sergey Shabalin
#Calculating distance to detected objects using depth map 

 # 1. Camera calibration: correcting camera lens distortion using OpenCV chessboard algorithm  (./chessboard/*.jpg)
 # 2. Rectifying two frames from camera using rotation matrix. ('./images/*.jpg')
 # 3. Disparity calculation and filtering (stereo vision effect)
 # 4. Threshold alpha-filter 
 # 5. Finding contours of objects and boxes centers
 # 6. Calculating distance to objects using Depth map, focal length and baseline. 
 # 7. Drawing contours and distances.

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
#
from calibration import CalibrateCamera
from rectifying import GetUndistortedImage, SaveUndistortedImage
from depth_calculating import GetDisparityMap, FindAnObstacle, CalculateDistance, ImagePutDistance

#for 5,6: 210-220
#for 3,4: 160-220

#Initializing images paths

image_left_path ='./images/3.jpg'
image_right_path='./images/4.jpg'
images_for_calibrating_path= './chessboard/*.jpg'


if __name__== "__main__":
       
    # We need to get distortion matrix from camera calibrating procedure
    # DistortionMatrix = (ret,mtx,dist,rvecs,tvecs)
    DistortionMatrix = CalibrateCamera(images_for_calibrating_path)
    
    #Correcting camera lens distortion using distiorion matrix from previous step
    #Getting the rectified pair 
    
    imgR_rectified = GetUndistortedImage(image_right_path,DistortionMatrix)
    imgL_rectified = GetUndistortedImage(image_left_path, DistortionMatrix)
    
    #Getting left and right disparity maps for filtering
    
    disparity_left, disparity_right, filteredDisparity = GetDisparityMap(imgL_rectified, imgR_rectified)
        
    #Finding obstacle cetner coordinates
    object_centers = FindAnObstacle(filteredDisparity, imgL_rectified)

    #Calculating distance
    distance = CalculateDistance(object_centers,disparity_left)

    #Drawing lines and distances     
    image = ImagePutDistance(imgL_rectified, object_centers, distance)
    
    #Showing images 
    cv2.imshow('object1',image)
    cv2.imshow('Filtered_disparity',filteredDisparity)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows
    
    