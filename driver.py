from mtcnn.detector import detector
from blur import pixelate
from PIL import Image
import cv2
import numpy as np  
import time

SANITY_FLIP = 0
MAX_SANITY_FLIP = 5
start_time = None
counter = None

def sanity_check(prev, boxes):
    """sanity check function. helps to reduce edge case falses.check docs
       for more info.
       
       Parameters:
       -> prev: list of bounding boxes in the previous frame
       -> boxes: list of bounding boxes in the current frame

       Returns:
       -> updated prev and boxes (respectively)
    """
    
    global SANITY_FLIP
  

    if len(prev) == 0:
        return boxes, boxes
    if len(boxes) == 0:
        if SANITY_FLIP < MAX_SANITY_FLIP :
            SANITY_FLIP += 1
            prev[:,0] += 20
            prev[:,1] += 20
            prev[:,2] += 20
            prev[:,3] += 20
            return prev, prev
    
    if len(prev) == 0 and len(boxes) == 0:
        return [], []
    SANITY_FLIP = 0

    _boxes = boxes
    _prev = []
    prev, boxes = prev[:,:4], boxes[:,:5]
    boxes[:,4] = (boxes[:,2]-boxes[:,0]+1.0)*(boxes[:,3]-boxes[:,1]+1.0)

    last = len(boxes)
    to_add = []
    for idx, pi in enumerate(prev):
        
        x1, y1, x2, y2 = [pi[i] for i in range(4)]
        area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    
        x1 = np.maximum(x1, boxes[:last, 0])
        y1 = np.maximum(y1, boxes[:last, 1])
        x2 = np.minimum(x2, boxes[:last, 2])
        y2 = np.minimum(y2, boxes[:last, 3])

        inter_area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)

        inter_area = inter_area.reshape((-1,1))
        overlap = inter_area/(boxes[:,4].reshape((-1,1)) + area - inter_area)  
        

        indices = np.where(overlap > .2)[0]

        if len(indices) == 0:

            second_check = np.where(overlap > .1)[0]
            if len(second_check) != 0:
                _prev.append(np.append(prev[idx].reshape((1,-1)), np.array([0,0,0,0,0]).reshape((1,-1))))

    if len(_prev) == 0:       
        return _boxes, _boxes #prev, boxes
    
    _prev = np.array(_prev)
    _boxes = np.vstack((_boxes, _prev))
    return _boxes,_boxes   


def pixelate(img, b):
    """
    pixelates faces inside bounding boxes.

    Parameters:
    -> img: the frame on which faces have to be blurred
    -> b: list of bounding boxes of the shape [n, 9]
    """

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

def draw_boxes(frame, prev, boxes, draw_prev = False):
    """
    utility function to draw boxes(previous and current) around faces.
    Primarily used for debugging purposes.

    Parameters:
    -> frame (PIL or array): The image on which the boxes have to be drawn
    -> prev (numpy.ndarray): array of bounding boxes for the previous frame
    -> boxes (numpy.ndarray): array of bounding boxes for the the current frame
    -> draw_prev (bool): draws previous frame's bounding boxes if True

    """
    
    if len(boxes) == 0:
        return frame
    if isinstance(prev, tuple):
        return frame
    x1, y1, x2, y2 = [boxes[:, i].reshape(-1,1) for i in range(4)]
    if draw_prev:
        _x1, _y1, _x2, _y2 = [prev[:, i].reshape(-1,1) for i in range(4)]
    for a, b, c, d in zip(x1, y1, x2, y2):
        cv2.rectangle(frame, (a,b), (c,d), color=(0,0,255), thickness=2)
    if draw_prev:
        for a, b, c, d in zip(_x1, _y1, _x2, _y2):
            cv2.rectangle(frame, (a,b), (c,d), color=(255,0,255), thickness=1)
    return frame    
        
    

def driver_func(path, targpath = None, view = True, _max = -1):
    """
    primary driver function to perform the pixelation

    Parameters: 
    -> path: source of the video file
    -> targpath: location to save the operated video in. Only '.avi' 
                 extension supported
    -> view: shows realtime pixelation if True
    -> _max: maximum number of frames to process . Indefinitely processes if _max=-1
             until manually quit
    """

    if targpath is not None:
        frame_list = []
    global start_time, counter
    
    cap = cv2.VideoCapture(path)
    count = 0
    
    prev = np.asarray([])
    width, height, _ = cap.read()[1].shape
    #fps = cap.get(cv2.)
    #print("FPS:", fps)
    start_time = time.time()
    while cap.isOpened():
        
        _, frame = cap.read()
        try:
            b = detector(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except:
            b = []
            pass

        count += 1
        
        if isinstance(b, tuple):
            pass
        else:
            prev, b=sanity_check(prev, b)
            frame = pixelate(frame, b)
        
        if targpath is not None:
            frame_list.append(frame)
        cv2.imshow("Frame", frame)
        
        if count == _max:
            counter = count
            break
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            counter = count
            break
        if key == ord('p'):
            if cv2.waitKey() == ord('p'):
                continue
    if targpath is not None:
        save_video(frame_list, height, width, targpath)      
    print("Time Taken to process %d frames is: %.2f"%(count, time.time()-start_time))


def save_video(frame_list, h, w, path):
    """
    helper function to save the video

    Parameters:
    -> frame_list: list of frames to be converted into a video
    -> h : height of original video
    -> w : width of original video
    """
    h = int(h)
    w = int(w)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')    
    video_tracked = cv2.VideoWriter(path, fourcc=fourcc, frameSize=(h,w), fps=25.0)
    for frame in frame_list:
        video_tracked.write(frame)
    video_tracked.release()

    return 

"""
path = "tests/videotest2.mp4"
driver_func(path, targpath="tests/test1.avi", _max=500)
"""