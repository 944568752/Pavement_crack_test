# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:01:17 2021

@author: 94456
"""


# Crack segmentation



import warnings
warnings.filterwarnings('ignore')


import os


import numpy as np
import cv2 as cv


file_path=r'./test_2.png' 


def histeq3(imarr2):
    # https: // blog.csdn.net / lzwarhang / article / details / 93304644
    
    hist, bins = np.histogram(imarr2, 255)
    cdf = np.cumsum(hist)
    cdf = 255*(cdf/cdf[-1])
    imarr3 = np.interp(imarr2, bins[:-1], cdf)
    
    return imarr3


def otsu(img):
    h=img.shape[0]
    w=img.shape[1]
    m=h*w   # 图像像素点总和
    otsuimg=np.zeros((h,w),np.uint8)
    threshold_max=threshold=0   # 定义临时阈值和最终阈值
    histogram=np.zeros(256,np.int32)   # 初始化各灰度级个数统计参数
    probability=np.zeros(256,np.float32)   # 初始化各灰度级占图像中的分布的统计参数
    for i in range (h):
        for j in range (w):
            s=img[i,j]
            histogram[s]+=1   # 统计像素中每个灰度级在整幅图像中的个数
    for k in range (256):
        probability[k]=histogram[k]/m   # 统计每个灰度级个数占图像中的比例
    for i in range (255):
        w0 = w1 = 0   # 定义前景像素点和背景像素点灰度级占图像中的分布
        fgs = bgs = 0   # 定义前景像素点灰度级总和and背景像素点灰度级总和
        for j in range (256):
            if j<=i:   # 当前i为分割阈值
                w0+=probability[j]   # 前景像素点占整幅图像的比例累加
                fgs+=j*probability[j]
            else:
                w1+=probability[j]   # 背景像素点占整幅图像的比例累加
                bgs+=j*probability[j]
        u0=fgs/w0   # 前景像素点的平均灰度
        u1=bgs/w1   # 背景像素点的平均灰度
        g=w0*w1*(u0-u1)**2   # 类间方差
        if g>=threshold_max:
            threshold_max=g
            threshold=i
    print(threshold)
    for i in range (h):
        for j in range (w):
            if img[i,j]>threshold:
                otsuimg[i,j]=255
            else:
                otsuimg[i,j]=0
    return otsuimg




original_image=cv.imread(file_path)

gray_image=cv.cvtColor(original_image,cv.COLOR_BGR2GRAY)


# cache=np.array([256],dtype=np.uint8)
# gray_image=cache-gray_image



# Gaussian blur image
Gaussian_image=cv.GaussianBlur(gray_image,(5,5),0)
# Gaussian_image=cv.medianBlur(gray_image,9)

# Gaussian_image=cv.blur(gray_image,(7,7))


Gaussian_image=cv.equalizeHist(Gaussian_image)

cv.imwrite('./Gaussian_image.png',Gaussian_image)



# Laplacian_image=cv.Laplacian(Gaussian_image,cv.CV_16S,ksize=3)
# Laplacian_image=cv.convertScaleAbs(Laplacian_image)


# gradX = cv.Sobel(Gaussian_image, ddepth=cv.CV_32F, dx=1, dy=0)
# gradY = cv.Sobel(Gaussian_image, ddepth=cv.CV_32F, dx=0, dy=1)

# gradient = cv.subtract(gradX, gradY)
# Laplacian_image = cv.convertScaleAbs(gradient)



# Laplacian_image=otsu(Gaussian_image)


ret,Laplacian_image=cv.threshold(Gaussian_image,200,255,cv.THRESH_BINARY)


cv.imwrite('./test_result.png',Laplacian_image)


