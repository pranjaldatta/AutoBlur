from mtcnn.detector import detector
from blur import pixelate
from PIL import Image
import cv2
import numpy as np  

def sanity_check(prev, boxes):
    """sanity check goes in here"""


def pixelate(img, b):

    if len(b) == 0:
        return img
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    x1, y1, x2, y2 = [b[:,i].reshape(-1, 1) for i in range(4)]
    
    for x1, y1, x2, y2 in zip(x1, y1, x2, y2):
        
        x1, y1, x2, y2 = int(x1[0]), int(y1[0]), int(x2[0]), int(y2[0])
        face = img.crop((x1, y1, x2, y2))
        _face = face.resize((8,8), resample=Image.BILINEAR)
        face = _face.resize(face.size, resample=Image.NEAREST)
        img.paste(face, box=(x1, y1, x2, y2))
    
    return np.array(img)

def draw_boxes(frame, prev, boxes):
    
    if len(boxes) == 0:
        return frame
    
    x1, y1, x2, y2 = [boxes[:, i].reshape(-1,1) for i in range(4)]
    for x1, y1, x2, y2 in zip(x1, y1, x2, y2):
        cv2.rectangle(frame, (x1,y1), (x2,y2), color=(0,0,255), thickness=1)
    return frame    
        
    

def driver_func(path):
    cap = cv2.VideoCapture(path)
    count = 0
    while cap.isOpened():
        
        _, frame = cap.read()
        if count < 400:
            count += 1
            continue

        b = detector(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if not isinstance(b, tuple):          
            b = np.array(b)
            frame = pixelate(frame, b)

        cv2.imshow("Frame", frame)
    
        if cv2.waitKey(1) == ord('q'):
            break
    


path = "videotest2.mp4"
driver_func(path)