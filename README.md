# AutoBlur
Blur faces in a video automatically

### Current bluring performance
works well for most but not well enough in edge cases.also realtime reduces frame rate considerably. dont know about saving. 

### to do:
save the video after blur

### Case for sanity check 
MTCNN detects faces on a frame by frame basis. While mtcnn is generally very fast, performant and accurate, there arises many problems when dealing with edge cases. For
example, when someone turns his face from the camera across frames:  what happens is threre comes a transition point wherein the MTCNN doesnt detect a face but a human eye can easily identify the face so that kinda beats the purpose. So the proposed sanity check measure is this:

given two frames: i-1 and i:
we calculate the IoU of boxes in ith frame vs boxes of (i-1)th frame. If for a given box in (i-1)th frame, corresponding overlap wrt all boxes in ith are below a certain threshold then we include the given i-1th bounding box also in the bounding box list of the ith frame
