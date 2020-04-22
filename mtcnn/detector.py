import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from nets import PNet, RNet
from mtcnn.stage_one import first_stage
from mtcnn.stage_two import get_image_boxes
from utils.visualize import show_boxes
from utils.utils import nms, convert_to_square


def detector(image, min_face_size = 20, threshold = 0.7 ):
    
    pnet = PNet()
    rnet = RNet()

    h, w = image.size
    min_length = min(h, w)
    min_detection_size = 12
    
    scale_factor = 0.709   #not sure why its .709
    scales = []
    m = min_detection_size/min_face_size
    min_length *= m
    factor_count = 0
    
    while min_length >= min_detection_size:
        scales += [m * np.power(scale_factor,factor_count)]
        min_length *= scale_factor
        factor_count += 1
   
    ################## Stage 1 #############################

    bounding_boxes = []

    for s in scales:
        boxes = first_stage(image,s,pnet,.8)
        bounding_boxes.append(boxes)   
    #bounding_boxes has shape [n_scales, n_boxes, 9]
    
    #remove those scales for which bounding boxes were none
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    
    #Add all the boxes for each scale 
    bounding_boxes = np.vstack(bounding_boxes)  # returns array of shape [n_boxes, 9]
   
    im = show_boxes(image,bounding_boxes)
    im.show()


    ################## Stage 2 ########################
    
    img_box = get_image_boxes(bounding_boxes,image,size=24)
    
    img_box = torch.tensor(img_box, requires_grad=True)

    probs,boxes = rnet(img_box)
    #print("After rnet: ")
    #print(probs)

    probs = probs.data.numpy() #Shape [boxes, 2]
    boxes = boxes.data.numpy() #Shape [boxes, 4]
    

    ind = np.where(probs[:, 1] > 0.7)

    print("Acceptable boxes: ", len(ind[0]))
    
    bounding_boxes = bounding_boxes[ind]
    bounding_boxes[:, 4] = probs[ind, 1].reshape((-1,))
    #boxes = boxes[ind]
    
    keep = nms(bounding_boxes)
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = convert_to_square(bounding_boxes)

    print('Number of bounding boxes after stage 2: ', len(bounding_boxes))

    im1 = show_boxes(image, bounding_boxes)
    im1.show()
    
    return bounding_boxes

image = Image.open("/Users/sashrikasurya/Documents/AutoBlur/test.jpg")    
b = detector(image)    

