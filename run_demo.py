# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#
#Modified by Mayank Sharma
#sharma.mayank2125@gmail.com
#2019/12/05

import numpy as np
from affine_ransac import Ransac
from align_transform import Align
from affine_transform import Affine
import glob
import cv2
from scipy.signal.signaltools import correlate2d as c2d
from scipy.stats import norm
from multiprocessing import Pool, TimeoutError
from itertools import product
import argparse
import os



# Affine Transform
# |x'|  = |a, b| * |x|  +  |tx|
# |y'|    |c, d|   |y|     |ty|
# pts_t =    A   * pts_s  + t

# -------------------------------------------------------------
# Test Class Affine
# -------------------------------------------------------------

# Create instance
af = Affine()

# Generate a test case as validation with
# a rate of outliers
outlier_rate = 0.9
A_true, t_true, pts_s, pts_t = af.create_test_case(outlier_rate)

# At least 3 corresponding points to
# estimate affine transformation
K = 3
# Randomly select 3 pairs of points to do estimation
idx = np.random.randint(0, pts_s.shape[1], (K, 1))
A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

# Display known parameters with estimations
# They should be same when outlier_rate equals to 0,
# otherwise, they are totally different in some cases
# print(A_true, '\n', t_true)
# print(A_test, '\n', t_test)

# -------------------------------------------------------------
# Test Class Ransac
# -------------------------------------------------------------

# Create instance
rs = Ransac(K=3, threshold=1)

residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)

# Run RANSAC to estimate affine tansformation when
# too many outliers in points set
A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)
# print(A_rsc, '\n', t_rsc)

# -------------------------------------------------------------
# Test Class Align
# -------------------------------------------------------------

al = Align(threshold=1)

def drawContours(image,bin_roi):

    image = cv2.imread(image)
    image = al.image_resize(image,height=900)
   
    bin_roi = bin_roi.astype('uint8')
    bin_roi = cv2.cvtColor(bin_roi, cv2.COLOR_BGR2GRAY)

    

    imagess, contours, hierarchy = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 0, 255), 1) 
  
    cv2.imshow('Contours', image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

def compareAndDetect(good_file,bad_file):

    diff_img = al.align_image(good_file, bad_file)
    # print (diff_img)
    if (diff_img is None):
        return None

    else :
        return diff_img/255

    
def listOfTuples(l1, l2): 
    return list(map(lambda x, y:(x,y), l1, l2)) 

def findAggregratedComparision(good_folders,defectFile):

    next_img = None

    images = [item for sublist in [glob.glob(good_folders + ext) for ext in ["/*.jpeg", "/*.jpg","/*.tif","/*.png"]] for item in sublist]


    
    bad_file = defectFile


    args_imgs = listOfTuples(images,[bad_file]*len(images))
    # print(args_imgs)
    # print (arg)
    with Pool(processes = len(images)) as pool:

        results = pool.starmap(compareAndDetect,args_imgs)


    # results_filtered = results.remove(None)
    results = [res for res in results if res is not None]
    # print (results)
    image_counter = len(results)

    if image_counter==0:

        print ("-----------------")
        print ("BAD SAMPLE")
        print ("----------------")

        return
    # print (results)
    final_res = np.zeros_like(results[0])
    

    for res in results:
                final_res = final_res+res
          


    next_img = final_res
    # print (image_counter)
    next_img[next_img < (image_counter*0.9)] = 0
    next_img[next_img > (image_counter*0.9)] = 1
    next_img = next_img*255


    drawContours(bad_file,next_img)

    print ("-------------------")

    if np.count_nonzero(next_img) > 0: # if the combined difference mask contains any positve pixel,mark it as defective,try changing this threshold to relax defect tolerance
        print ("BAD SAMPLE.")
    else:
        print ("GOOD SAMPLE")

    print ("---------------------")

     




if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Paths to good folders and current image')
    parser.add_argument('goodImagesDir', type=str, help='Input dir for good samples')
    parser.add_argument('BadImagePath', type=str, help='Image which needs to be checked')

    args = parser.parse_args()

    good_folders = args.goodImagesDir
    defectFile = args.BadImagePath

    findAggregratedComparision(good_folders,defectFile)

