from mtcnn.detector import detector
from PIL import Image
import cv2
import numpy as np

#path ="tests/test5.jpg"

#img = Image.open(path)
#b = detector(img)

def pixelate(img, b):

    if len(b) == 0:
        return img

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    for bi in b:
        if len(bi) == 0:
            continue
        
        x1, y1, x2, y2 = [int(bi[i]) for i in range(4)]
        face = img.crop((x1,y1,x2,y2))
        face1 = face.resize((8,8), resample=Image.BILINEAR)
        face = face1.resize(face.size, resample=Image.NEAREST)
        img.paste(face, box=(x1,y1,x2,y2))
    return img
#img.show()