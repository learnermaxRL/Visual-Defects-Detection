# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/11
#


import cv2
import numpy as np
from affine_ransac import Ransac
from affine_transform import Affine
from scipy.signal.signaltools import correlate2d as c2d
import math
from scipy.stats import norm
# from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_ssim
import gauss
from scipy import signal
from scipy import ndimage

# The ration of the best match over second best match
#      distance of best match
# ------------------------------- <= MATCH_RATIO
#  distance of second best match
RATIO = 0.8



class Align():

    def __init__(self,
                 K=3, threshold=1):
        ''' __INIT__

            Initialize the instance.

            Input arguments:

            - source_path : the path of sorce image that to be warped
            - target_path : the path of target image
            - K : the number of corresponding points, default is 3
            - threshold : a threshold determins which points are outliers
            in the RANSAC process, if the residual is larger than threshold,
            it can be regarded as outliers, default value is 1

        '''


        self.K = K
        self.threshold = threshold

        # uncomment and provide calib files in case you wish to undistort images.
        # self.mtx = np.load('../callibratedFiles/mtx.npy')
        # self.dist = np.load('../callibratedFiles/dist.npy')

    def read_image(self, path, mode=1):
        ''' READ_IMAGE

            Load image from file path.

            Input arguments:

            - path : the image to be read
            - mode : 1 for reading color image, 0 for grayscale image
            default is 1

            Output:

            - the image to be processed

        '''

        return cv2.imread(path, mode)

    def extract_SIFT(self, img):
        ''' EXTRACT_SIFT

            Extract SIFT descriptors from the given image.

            Input argument:

            - img : the image to be processed

            Output:

            -kp : positions of key points where descriptors are extracted
            - desc : all SIFT descriptors of the image, its dimension
            will be n by 128 where n is the number of key points


        '''

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract key points and SIFT descriptors
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(img_gray, None)

        # Extract positions of key points
        kp = np.array([p.pt for p in kp]).T

        return kp, desc

    def noise_removal(self,img):

        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

        # mask = np.dstack([mask, mask, mask]) / 255
        # out = img * mask

        return mask

    def getdistanceKP(self,img1,img2,kp1,kp2):

            dist=[]
            angles = []

            rows1 = img1.shape[0]
            cols1 = img1.shape[1]
            rows2 = img2.shape[0]
            cols2 = img2.shape[1]

            # print (kp1)
            # print (type(kp1))
            # print (len(kp1))

            no_of_matches =len(kp1[0])
            p1 = np.zeros((no_of_matches, 2)) 
            p2 = np.zeros((no_of_matches, 2))
         
            for i in range(no_of_matches):
                  # print(type(matches[i])) 
                    p1[i, :] = (kp1[0][i],kp1[1][i]) 
                    p2[i, :] = (kp2[0][i],kp2[1][i]) 

                    angle =  np.rad2deg(np.arctan2(p1[i][1] - p2[i][1], p1[i][0] - p2[i][0]))
                    eudistance =math.sqrt(math.pow(p1[i][0]-(p2[i][0]+cols1),2) + math.pow(p1[i][1]-p2[i][1],2) )



                    dist.append(eudistance)
                    angles.append(angle)
            
            # print (dist)
            # print (angles)


            mu_dist, std_dist = norm.fit(dist)
            mu_angles, std_angles = norm.fit(angles)

          # appying angle filter fist ---> distance_filter to get filtered kps

            
            dist_filter = np.squeeze(np.argwhere(abs(dist-mu_dist)<(0.7*std_dist))).tolist()
          
            angle_filter = np.squeeze(np.argwhere(abs(angles-mu_angles)<(0.6*std_angles))).tolist()
          
            common_indices = [] 
            for item in dist_filter:
                   if item in angle_filter and item not in common_indices :
                       common_indices.append(item) 


            new_kp1 = np.zeros(shape = (2,len(common_indices)))  
            new_kp2 = np.zeros(shape = (2,len(common_indices)))

            for c,i  in enumerate(common_indices):
                new_kp1[:,c] = kp1[:,i]
                new_kp2[:,c] = kp2[:,i]

            return new_kp1,new_kp2

    def ssim(self,img1, img2, cs_map=False):
  
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        size = 11
        sigma = 1.5
        window = gauss.fspecial_gauss(size, sigma)
        K1 = 0.01
        K2 = 0.03
        L = 255 #bitdepth of image
        C1 = (K1*L)**2
        C2 = (K2*L)**2
        mu1 = signal.fftconvolve(window, img1, mode='valid')
        mu2 = signal.fftconvolve(window, img2, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
        sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
        sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
        if cs_map:
            return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2)), 
                    (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
        else:
            return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2))

    def msssim(self,img1, img2):
        
            level = 5
            weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
            downsample_filter = np.ones((2, 2))/4.0
            im1 = img1.astype(np.float64)
            im2 = img2.astype(np.float64)
            mssim = np.array([])
            mcs = np.array([])
            for l in range(level):
                ssim_map, cs_map = self.ssim(im1, im2, cs_map=True)
                mssim = np.append(mssim, ssim_map.mean())
                mcs = np.append(mcs, cs_map.mean())
                filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                        mode='reflect')
                filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                        mode='reflect')
                im1 = filtered_im1[::2, ::2]
                im2 = filtered_im2[::2, ::2]
            return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                        (mssim[level-1]**weight[level-1]))
    def mse(mse,imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def match_SIFT(self, desc_s, desc_t):
        ''' MATCH_SIFT

            Match SIFT descriptors of source image and target image.
            Obtain the index of conrresponding points to do estimation
            of affine transformation.

            Input arguments:

            - desc_s : descriptors of source image
            - desc_t : descriptors of target image

            Output:

            - fit_pos : index of corresponding points

        '''

        # Match descriptor and obtain two best matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)

        # Initialize output variable
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        for i in range(matches_num):
            # Obtain the good match if the ration id smaller than 0.8
            if matches[i][0].distance <= RATIO * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx,
                                 matches[i][0].trainIdx])
                # Put points index of good match
                fit_pos = np.vstack((fit_pos, temp))

        # print (np.count_nonzero(fit_pos))
        # print ("ooooooooooooo")
        return fit_pos

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        ''' AFFINE_MATRIX

            Compute affine transformation matrix by corresponding points.

            Input arguments:

            - kp_s : key points from source image
            - kp_t : key points from target image
            - fit_pos : index of corresponding points

            Output:

            - M : the affine transformation matrix whose dimension
            is 2 by 3

        '''

        # Extract corresponding points from all key points
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]

        # Apply RANSAC to find most inliers
        _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)

        # Extract all inliers from all key points
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]

        A, t = Affine().estimate_affine(kp_s, kp_t)

        M = np.hstack((A, t))

        return M



    def undistort(self,img):

    # img = image_resize(img,height=720)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            # undistort
            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            # dst = image_resize(dst,height=72)
            return dst


    def check_linear_transofm_components(self,M):

        check_passed = False

        scaleX= math.sqrt(M[0][0]**2 + M[0][1]**2)
        scaleY= math.sqrt(M[1][0]**2 + M[1][1]**2)
        tx = M[0][2]
        ty = M[1][2]
        roation = (math.atan(M[1][0]/M[0][1]))
        shaer = (M[0][1]+M[1][0])/M[0][0]


        if ((scaleX < 1.2 and scaleX >0.9 ) and (scaleY < 1.2 and scaleY >0.9 )) and (abs(tx) < 70 and abs(ty)<70):
            check_passed = True


        return check_passed,scaleX,scaleY,roation,tx,ty


        




    def warp_image(self, source, target, M):
        ''' WARP_IMAGE

            Warp the source image into target with the affine
            transformation matrix.

            Input arguments:

            - source : the source image to be warped
            - target : the target image
            - M : the affine transformation matrix

        '''

        # Obtain the size of target image
        rows, cols, _ = target.shape

        # Warp the source image
        valid,scaleX,scaleY,roation,tx,ty = self.check_linear_transofm_components(M)


        if not valid:
            return None



        warp = cv2.warpAffine(source, M, (cols, rows))

        added_image = cv2.addWeighted(warp,1,target,1,0)
        sub_image = cv2.subtract(warp,target)

        ret,thresImg = cv2.threshold(sub_image,80,255,cv2.THRESH_BINARY)


        return sub_image

    def image_resize(self,image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized



    def undistort(self,img):

        # img = image_resize(img,height=720)
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
        # undistort
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        # dst = image_resize(dst,height=72)
        return dst



    def align_image(self,source_path,target_path):
        ''' ALIGN_IMAGE

            Warp the source image into target image.
            Two images' path are provided when the
            instance Align() is created.

        '''

        # Load source image and target image
        img_source = self.read_image(source_path)
        img_target = self.read_image(target_path)

        img_source = self.image_resize(img_source,height=900)
        img_target = self.image_resize(img_target,height=900)

       
        # img_source = self.undistort(img_source)
        # img_target = self.undistort(img_target)

        # Extract key points and SIFT descriptors from
        # source image and target image respectively
        kp_s, desc_s = self.extract_SIFT(img_source)
        kp_t, desc_t = self.extract_SIFT(img_target)

        # Obtain the index of correcponding points
        fit_pos = self.match_SIFT(desc_s, desc_t)

        # Compute the affine transformation matrix
        M = self.affine_matrix(kp_s, kp_t, fit_pos)


        # print  (M)

        # Warp the source image and display result
        diff= self.warp_image(img_source, img_target, M)
        if diff is None:
            return None
# 



        ret,bin_img = cv2.threshold(diff,100,255,cv2.THRESH_BINARY)

        return  bin_img
