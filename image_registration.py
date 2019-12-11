# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#


import numpy as np
from affine_ransac import Ransac
from align_transform import Align
from affine_transform import Affine
import glob
import cv2
from scipy.signal.signaltools import correlate2d as c2d
from scipy.stats import norm



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

# Load source image and target image

al = Align(threshold=1)

def findAggregratedComparision(good_folders,defectFile):

    next_img = None



    images = glob.glob(good_folders+'/*.tif')
    bad_file = defectFile
    image_counter = 0

    for image in images:
        # print (image)

        if image == bad_file:
            continue
        # al = Align(image, bad_file, threshold=1)
        # print (image)

    # Image transformation
        
        diff_img = al.align_image(image, bad_file)
        if (diff_img is None):
           continue
        # if score < 0.07 :
        #     continue
        
     
        # print (image)
        if next_img is not None:

            # diff_img =  cv2.add(diff_img, next_img)
            next_img = (diff_img)/255+next_img
        else:

            next_img = diff_img/255

        image_counter = image_counter+1


    # print (next_img)

    next_img[next_img < (image_counter*0.8)] = 0
    next_img[next_img > (image_counter*0.8)] = 1
    next_img = next_img*255

    cv2.imshow('img_commonssss', next_img)
    cv2.waitKey(0) 

if __name__ == "__main__":
    findAggregratedComparision(good_folders,defectFile)

