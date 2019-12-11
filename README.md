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
  
 **Output**

The code will output an image which will contain red regions indicating that their is an anomaly present,more often than not this may also contain regions of anomaly(the areas which didnt match),however in case of very apparent defects it may show more regions since it couldnt match properly.Apart from this it will output the "good or bad" in console.
In other case it can directly output "bad image" if the model is unable to find a suitable match for the current image in good folder

**Sample Detections**
![sample detection1 ](https://github.com/learnermaxRL/AnomalyDetection/blob/master/media/def1.png)
![sample detection2 ](https://github.com/learnermaxRL/AnomalyDetection/blob/master/media/def3.png)
![sample detection3 ](https://github.com/learnermaxRL/AnomalyDetection/blob/master/media/def6.png)

