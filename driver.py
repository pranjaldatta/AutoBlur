from mtcnn.detector import detector
#from blur import pixelate
from PIL import Image
import cv2
import numpy as np  

def sanity_check(prev, boxes):
    """sanity check goes in here. working in it"""


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
    w,h,_ = cap.read()[1].shape
    count = 0
    frames_tracked = []
    while cap.isOpened():
        
        _, frame = cap.read()

        #w,h,_ = frame.shape
        
        if count < 100: #just for the current video
            count += 1
            continue

        b = detector(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        if not isinstance(b, tuple):          
            b = np.array(b)
            frame = pixelate(frame, b)
            
        frames_tracked.append(frame)
        cv2.imshow("Frame", frame)
    
        if cv2.waitKey(1) == ord('q'):
            break
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')    
    video_tracked = cv2.VideoWriter('video_tracked.avi', fourcc, 20.0, (h,w))
    for frame in frames_tracked:
        video_tracked.write(frame)
    video_tracked.release()


path = "news_test.mp4"
driver_func(path)
