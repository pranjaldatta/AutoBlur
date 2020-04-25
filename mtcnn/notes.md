### stage 2
Basically what happens is that after the pnet + nms round, we have a set of bounding boxes,now we take the image inside the bounding box i.e. crop the image out of the original box adn feed it through rnet. rnet gives probabilities and "box offsets" for each such "proposal crop". "box offsets" are basically by how much should the original box be offseted such that it finds a face and its probabilities. after performing the usual elimination by confidence thresholding and nms, we pass it on to calibrate boxes.

### calibrate boxes(bouding_boxes, offsets)
basically offsets the original boxes by an amount as predicted by the rnet

