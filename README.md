# AnomalyDetection
This repo contains code to detect anonmaly by comparing an image with its golden refrences/template matching,originally working on automotive parts,but should be able to generalize well accross other domains.
The experimentation and results were done in controlled environment since it is meant to be used in industrial applications.

**controlled environment in this case include.**

- similar lighting setup accross all the images
- similar positional placement
- similar perspective of images

**Please go through sample images to get an idea of how the images should look like**

**Requirements**

- Numpy
- Opencv,Opencv-contrib 
- Matplotlib
- Scikit-Image,Scipy


**How to run**

python run_demo.py <path to good samples folder> <path to image which is to be analyzed for anomaly>
(use -h flag for help on args)

**Sample Detections**
![sample detection1 ](https://github.com/learnermaxRL/AnomalyDetection/blob/master/media/def1.png)
![sample detection2 ](https://github.com/learnermaxRL/AnomalyDetection/blob/master/media/def3.png)
![sample detection3 ](https://github.com/learnermaxRL/AnomalyDetection/blob/master/media/def6.png)

