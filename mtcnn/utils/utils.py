import numpy as np 
from PIL import Image
import torch

def nms(boxes, overlap_thresh = .5, mode='union'):
    """
    An utility function that performs nms over the bounding box

    Params:
    -> boxes: the bounding box proposals
    -> overlap_thresh: maximum permissible overlap ratio
    -> mode: default - union (IoU)

    Output:
    -> bounding box list with overlapping boxes removed
    """

    if len(boxes) == 0:
        return []

    x1, y1, x2, y2, confidence = [boxes[:,i] for i in range(5)]

    selected = [] #selected indices
    ids_sorted = np.argsort(confidence) #sorting in ascening order. returns indices
    areas = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)

    while(len(ids_sorted) > 0):
        """
        we loop through the sorted ids. 
        1. select the last id
        2. compare the chosen bbox IoU with all the others
        3. del the ones above the threshold.
        4. return selected ids
        """
        last_idx = len(ids_sorted) - 1
        selected.append(last_idx) #since max confidence
        del_index = [last_idx]

        #point to note: (0,0) is at top left of display
        #top left cornet of intersection area
        xi1 = np.maximum(x1[last_idx], x1[:last_idx])
        yi1 = np.maximum(y1[last_idx], y1[:last_idx])

        #bottom right corner of intersection area
        xi2 = np.minimum(x2[last_idx], x2[:last_idx])
        yi2 = np.minimum(y2[last_idx], y2[:last_idx])

        inter_area = (xi2 - xi1 + 1.0)*(yi2 - yi1 + 1.0)
        overlap = inter_area/(areas[last_idx] + areas[ids_sorted[:last_idx]] - inter_area)

        del_index = np.concatenate([del_index, np.where(overlap > overlap_thresh)[0]])
        print("Performing nms: deleting {} boxes".format(len(del_index)))
        ids_sorted = np.delete(ids_sorted, del_index)

    print("Performed nms. Returning {} box indices".format(len(selected)))
    return selected #selected indices
    

def preprocess(img):
    """
    A utiity function that takes a numpy image array or PIL
    Image adn returns a tensor
    
    Input: 
        -> img: input image in array or PIL format
    Output:
        -> tensor    
    """
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    img = torch.tensor(img, dtype=torch.float32, requires_grad=False)
    img = img.permute(2,0,1)
    img = torch.unsqueeze(img, 0)
    img = (img - 127.5)*0.0078125
    return img
    