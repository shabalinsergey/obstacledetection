
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

#Disparity calculation

def GetDisparityMap(imgL,imgR):
       
    #Disparity settings
    window_size = 3
    min_disp = 0
    num_disp = 128-min_disp
    matcher_left = cv2.StereoSGBM_create(
        blockSize = 5,
        numDisparities = num_disp,
        minDisparity = min_disp,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 5,
        preFilterCap = 63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)

    # Filter
    lmbda = 80000 #org 80000
    sigma = 1.2 #org 1.2400,330

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print ('Calculating disparity...')
    
    disparity_left = matcher_left.compute(imgL, imgR) .astype(np.float32)
    disparity_left = np.int16(disparity_left)
    disparity_right = matcher_right.compute(imgR, imgL) .astype(np.float32)
    disparity_right = np.int16(disparity_right)
    
    #Filtering disparity map
    filteredDisparity = wls_filter.filter(disparity_left, imgL, None, disparity_right)
    filteredDisparity = cv2.normalize(
        src=filteredDisparity,
        dst=filteredDisparity,
        beta=1,
        alpha=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
        )
    
    return disparity_left, disparity_right, filteredDisparity

def FindAnObstacle(filteredDisparity, imgL):
    threshold_min = 160
    threshold_max= 220
    #Finding contours by applying threshold filter
    r, threshold = cv2.threshold(filteredDisparity, threshold_min, threshold_max, cv2.THRESH_BINARY)
    cv2.imshow('threshold',threshold)
    contours,hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Contours count
    count=0
    #Object centers array
    object_centers = []
    #Try to filter too small and too big objects 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >1000 and area < 100000:
            rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
            box = np.int0(box)
            cv2.drawContours(imgL,[box],-1,(255,0,0),2)
            center=rect[0]
            object_centers.append(center)
    return object_centers
    

def CalculateDistance(object_centers,disparity_left):
    #calculating distance using normalized to baseline focuslength
    distance=[]
    focuslength_normalized = 800
    for i in range(0,len(object_centers),1):
        y = int(object_centers[i][1])
        x = int(object_centers[i][0])
        dist = float(focuslength_normalized/disparity_left[y,x])
        distance.append(dist)
  
    return distance

def ImagePutDistance(imgL_rectified, object_centers, distance):
    
    for i in range(0,len(distance),1):
        
        dist=float('{:.3f}'.format(distance[i]))
        y = int(object_centers[i][1])
        x = int(object_centers[i][0])
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        output_image = cv2.putText(imgL_rectified, str(dist), (int(x),int(y)), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    return output_image