# A bit about how it works

## How it works?

A general workflow is:

1. The video is read on a frame by frame basis.

2. The frame is passed onto the pretrained  mtcnn which generates the bounding boxes
and returns the results

3. The 'pixelate' function pixelates all the faces inside their respective bounding boxes.

4. The pixelated faces replace the original faces for the given frame.

## Blurring performance without sanity check

Works well but often blurring/pixelation fails in edge cases as often in successive frames, (i<sup>th</sup> and (i+1)<sup>th</sup> frames
mtcnn fails to detect a face. (maybe the face was in transition or obstructed)

## Case for sanity check

The probability of face disappearing between frames (i+1)<sup>th</sup> and i<sup>th</sup> is very low. Hence it beats the purpose of blurring if the blurring for a given face fails at the next frame. Hence we propose a sanity check methodology to deal with these cases.

## Sanity Check

The proposed sanity check measure is this:

Given two consecutive frames (i-1)<sup>th</sup> and i<sup>th</sup> :-

We calculate the IoU of bounding boxes in i<sup>th</sup> frame vs boxes of (i-1)<sup>th</sup> frame. If for a given box in (i-1)<sup>th</sup> frame, corresponding overlap w.r.t. all boxes in i<sup>th</sup> is below a certain threshold then we include the given (i-1)<sup>th</sup> bounding box also in the bounding box list for i<sup>th</sup> frame.

*Note:*  If a given bounding box doesn't appear for more than a given number of frames (hyperparameter adjustible), then it's ditched from the bounding box list.
