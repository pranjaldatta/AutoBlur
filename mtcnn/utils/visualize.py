from PIL import ImageDraw

def show_boxes(img, bounding_boxes):

    im = img.copy()
    draw = ImageDraw.Draw(im)
    

    for i in bounding_boxes:
        draw.rectangle([
            (i[0],i[1]),
            (i[2],i[3])
            ], outline = 'white')

    return im
