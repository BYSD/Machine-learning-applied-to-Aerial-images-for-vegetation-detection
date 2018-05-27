%matplotlib inline
import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from __future__ import division
import glob

param ={}
# for example sensitivity=5/256 color values in range [0,255]
param["s"] = 5
# RGB  thershold
param["boundary"] = [[45, 90, 90], [75, 210, 210]]
param['filenames'] = glob.glob('code/*.JPG')
param['output'] = 'outputs'

def show(image):
    # Figure size in inches
    plt.figure(figsize=(7, 7))
    
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')

def find_biggest_contour(image):
    # Copy
    image = image.copy()
    #input, gives all the contours, contour approximation compresses horizontal,
    #vertical, and diagonal segments and leaves only their end points. For example,
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology.
    #It has as many elements as the number of contours.
    #we dont need it
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask 
    
def preprocessing(filename):
    # reading of the image
    img = cv2.imread(filename)
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # convert to grayscale
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 
    lower = param["boundary"][0]
    upper = param["boundary"][1]
    
    s = param['s']
    lower = np.array([color-s if color-s>-1 else 0 for color in lower], dtype="uint8")
    upper = np.array([color+s if color+s<256 else 255 for color in upper], dtype="uint8")

    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)
    # copy of the output
    output1 = output.copy()
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 since I said 0.
    blur = cv2.GaussianBlur(output1,(3,3),0)

    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(th2,cv2.MORPH_OPEN,kernel, iterations =1)
    #closing = cv2.morphologyEx(th2,cv2.MORPH_CLOSE,kernel, iterations = 3)
    
    sure_bg = cv2.dilate(th2,kernel,iterations=9)
    im2, contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_img = np.copy(img)
    labl=[]
    threshold=True
    while threshold==True:
        x, y = find_biggest_contour(im2)
        if cv2.contourArea(x)< 15000.:
            threshold = False
        #(xx,yy),radius = cv2.minEnclosingCircle(x)
        #center = (int(xx),int(yy))
        #radius = int(radius)
        #if radius<250:
            #bounding_img = cv2.circle(img,center,radius,(0,255,0),5)
        labl.append(y)
        im2 = im2 - y
    outputname  = filename.split('/')[1].split('.')[0]
    print(outputname)
    cv2.imwrite(param['output']+'/'+outputname+'.jpg',sum(labl))
    print('okay')
    #show(bounding_img)
    
    
def run():
    for filename in param['filenames']:
        preprocessing(filename)
        
run()
