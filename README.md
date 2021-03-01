# enemyTracker
Automatic object tracking in a video game using python and opencv

### Motivation

This is work in progress showing feature detection (by the Shi-Tomasi algorithm), point clustering, and object tracking (by the Lucas-Kanade algorithm).  Red dots mark detected features, blue rectangles that appear briefly show clusters of features, and green dots mark a point that is being tracked.  Eventually, I would like to use this type of image processing to let my code automatically control my character in a game and react to its environment (e.g., run from enemies, walk while avoiding obstacles, etc).

I learned what I needed to write this script from several different sources: see the credits section below for details.

### Installation

The code was written in Python 3.7 on Windows 10. You will need to first install the necessary modules (or you can download them from pypi.org):

pip install opencv-python 
pip install numpy 
pip install scipy

### Usage

Once the libraries are installed, just run the main code, e.g. by executing

python tracker.v1.py

from a command line, or start it from an IDE if you use one. This will read in a raw gameplay video file.  The input file name is currently set to "enemy-approaches-ext.mp4" but you will need to change it to the name of your won file.  The program will then run and process and display each frame.  The processed frames will also be saved to the video file "show-tracking.avi".  

You can watch the result here for a sample DS3 gameplay: https://youtu.be/aPmLtCADNvI

There are several parameters near the top of the python code that you can modify/optimize (and you may need to depending on the game):

inputVideo = name of input video file

outputVideo = name of output video file (for the moment, this needs to be a .avi file)
fps = frames per second in the ouptut video

fsca = an factor to scale down the input video frame size (may be needed if your video seems to be running slowly)

shortDistance = any points closer than this number of pixels are considered close for clustering purposes
minClusterSize = minimum number of points in a cluster to trigger the tracking algorithm
maxCorners = maximum allowed number of features found by the Shi-Tomasi algorithm

trackerWinSize  = (width,height) of the window used by the Lucas-Kanade tracking algorithm
trackerMaxLevel = number of pyramid levels for the Lucas-Kanade algorithm
trackerCriteria = this is used by the tracking algorithm, and you shouldn't have to change it
maxTrackerError = maximum allowed tracking error
maxTrackerKills = maximum allowed number of times a tracker can fail before it's "decomissioned"

subtractorHist = number of frames to use for background subtraction
subtractorThresh = threshold for background subtraction

The program currently gets its input from a video file, but could be modified relatively easily to read from a live game window (see, e.g., https://github.com/susurrus1/DesktopObjectDetection), although the game would have to be displayed in windowed mode in order for the python script to be able to draw on the game screen.

### Credits

I could not have done any of this without some fantastic resources I found online, and I want to give them proper credit and thanks here.

* For the background sutraction technique (although I ended up using KNN instead of MOG2, but it's the same principle): https://pysource.com/2018/05/17/background-subtraction-opencv-3-4-with-python-3-tutorial-32/
* For the feature detection: docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html
* For the optical flow tracking method: https://pysource.com/2018/05/14/optical-flow-with-lucas-kanade-method-opencv-3-4-with-python-3-tutorial-31/
