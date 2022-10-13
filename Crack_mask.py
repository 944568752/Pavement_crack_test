# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:11:49 2021

@author: 94456
"""


# Crack mask



import warnings
warnings.filterwarnings('ignore')



import os


from PIL import Image
import numpy as np


image_path=r'./test_result.png'
mask_path=r'./test_result_mask.png'


original_image=Image.open(image_path)
gray_image=original_image.convert('L')


mask=Image.open(mask_path)
mask=mask.convert('L')
mask=np.array(mask)

mask=np.where(mask>0,1,0)


gray_image=gray_image*mask
gray_image=Image.fromarray(gray_image).convert('RGB')


gray_image.save('./final_mask.png')


