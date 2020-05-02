from driver import driver_func

path = "tests/videos/video2.mp4" #path to test video
#driver_func(path, _max=250) #no saving

"""
to run a save demo comment out the previous example  and
uncomment these
"""
driver_func(path, targpath="tests/videos/blurred_vid2.avi", _max=400)

