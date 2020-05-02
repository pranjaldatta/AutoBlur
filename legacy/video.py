from mtcnn.detector import detector
from PIL import Image
import cv2
from blur import pixelate
import numpy as np


def sanityCheck(img, prev, b_pred):

    if prev is not None and len(b_pred) == 0:
        print(b_pred, end=" ")
        b_pred = prev
        b_pred[:,0] += 10
        b_pred[:,1] += 10
        b_pred[:,2] += 10
        b_pred[:,3] += 10
        print(b_pred)
        
        return b_pred
    
    elif prev is not None:
        print("performing area swap")
        prev_area = (prev[:,2] - prev[:,0] + 1.0)*(prev[:,3] - prev[:,1] + 1.0)
        pred_area = (b_pred[:,2] - b_pred[:,0] + 1.0)*(b_pred[:,3] - b_pred[:,1] + 1.0)
        if np.absolute(pred_area - prev_area) > 50:
            b_pred = prev
        return b_pred

    prev = b_pred
    
    return prev, b_pred

path = "/home/pranjal/Projects/AutoBlur/videotest2.mp4"
cap = cv2.VideoCapture(path)

count = 0
prev = None
while cap.isOpened():
    ret, frame = cap.read()

    if count <100:
        count += 1
        print(count)
        continue
    
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    try:
        b = detector(frame.copy())
    except:
        print("excetp")
        b = []
        pass
    
    prev, b = sanityCheck(frame.copy(), prev, b)

 
    frame = pixelate(frame, b) 
    
    cv2.imshow("frame", cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) == ord('q'):
        break

"""
path = "tests/class2.jpg"
#img = Image.open(path)
img = cv2.imread(path)
#cv2.imshow('frame', cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB))
#if cv2.waitKey() == ord('q'):
#   cv2.destroyAllWindows()
img = Image.fromarray(img)
img.show()
"""