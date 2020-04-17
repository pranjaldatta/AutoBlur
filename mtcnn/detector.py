import cv2
import numpy as np
from nets import PNet
from mtcnn.stage_one import first_stage
from PIL import Image
from utils.visualize import show_boxes

def detector(image):
    
    pnet = PNet()
    h, w = image.size
    min_length = min(h, w)
    
    scale_factor = 0.709   #not sure why its .709
    scales = []
    min_detection_size = 12
    min_face_size = 20
    m = min_detection_size/min_face_size
    min_length *= m
    factor_count = 0
    
    while min_length >= min_detection_size:
        scales += [m * np.power(scale_factor,factor_count)]
        min_length *= scale_factor
        factor_count += 1
    print("scales = ")
    print(scales)
    bounding_boxes = []

    for s in scales:
        bounding_boxes += [first_stage(image,s,pnet,.8)]    
    #bounding_boxes has shape [no. of scales, no. of boxes, 9]
    
    #remove those scales for which bounding boxes were none
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    
    #Add all th boxes for each scale 
    bounding_boxes = np.vstack(bounding_boxes)  # returns array of shape [no. of boxes, 9]

    print('Number of bounding boxes:', len(bounding_boxes))

    return bounding_boxes

image = Image.open("/Users/sashrikasurya/Documents/AutoBlur/test.jpg")    
b = detector(image)    
im = show_boxes(image,b)
im.show()

