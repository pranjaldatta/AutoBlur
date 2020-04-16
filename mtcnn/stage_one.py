import math
import numpy as np  
import torch
from PIL import Image
from utils.utils import preprocess, nms
import cv2



def scale_boxes(probs, boxes, scale, thresh=.8):
    """
    A method that takes in the outputs of pnet, probabilities and 
    box cords for a scaled image and returns box cords for the 
    original image.

    Params:
    -> probs: probilities of a face for a given bbox; shape: [a,b]
    -> boxes: box coords for a given scaled image; shape" [1, 4, a, b]
    -> scale: a float denoting the scale factor of the image
    -> thresh: minimum confidence required for a facce to qualify

    Returns:
    -> returns a float numpy array of shape [num_boxes, 9]
    """
    stride = 2
    cell_size = 12

    inds = np.where(probs > thresh)
    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [boxes[0, i, inds[0], inds[1]] for i in range(4)]  
    offsets = np.array([tx1, ty1, tx2, ty2])
    confidence = probs[inds[0], inds[1]]
    
    #no clue whats happening
    bboxes = np.vstack([
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        confidence,
        offsets
        ])
    return bboxes.T


def first_stage(img, scale, pnet, thresh=.8):
    """
    A method that accepts a PIL Image, 
    runs it through pnet and does nms.

    Params:
    -> img: PIL image
    -> scale: a float that determines the scaling factor
    -> pnet: an instance of the pnet
    -> thresh: threshold below which facial probs are unacceptable

    Returns:
    -> numpy array of type float of shape [num_boxes, 9]
       which contain box cords for a givens scale, confidence,
       and offsets to actual size
    """

    orig_h, orig_w = img.size
    scaled_h, scaled_w = math.ceil(scale*orig_h), math.ceil(scale*orig_w)
    img = img.resize((scaled_h, scaled_w), Image.BILINEAR)
    
    img = preprocess(img)
    probs, boxes = pnet(img)
    probs = probs.data.numpy()[0,1,:,:]
    boxes = boxes.data.numpy()

    bounding_boxes = scale_boxes(probs, boxes, scale)
    selected_ids = nms(bounding_boxes[:,0:5]) #indices to be kept
    
    return bounding_boxes[selected_ids]



    
    
"""
from nets import PNet
pnet = PNet()
first_stage(Image.open("/home/pranjal/Pictures/test.jpg"), .2,pnet,.8)
"""
