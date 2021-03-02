import numpy as np
import cv2
import scipy.cluster.hierarchy as hcluster

########################################################################################################################

# The parameters below should be adjusted/optimized by the user:

inputVideo = "enemy-approaches-ext.mp4"

outputVideo = "show-tracking.v2.avi"
fps = 30.0

fsca = 0.6

shortDistance = 20
minClusterSize = 8
maxCorners = 50

maxTrackers       = 10
trackerWinSize    = (15,15)
trackerMaxLevel   = 4
trackerCriteria   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
maxTrackerError   = 10.0
maxTrackerKills   = 3
trackerArrowScale = 10.0

subtractorHist = 4
subtractorThresh = 1000

########################################################################################################################

def pointDistance(pt1,pt2):
    x1,y1 = pt1.ravel()
    x2,y2 = pt2.ravel()

    return np.sqrt((x2-x1)**2+(y2-y1)**2)

class LKTracker:
    def __init__(self,oldPoints,winSize,maxLevel,criteria):
        self.oldPoints = oldPoints
        self.lkParams  = dict(winSize = winSize,maxLevel = maxLevel,criteria = criteria)
        # The kill count is used to eliminate trackers that don't perform well
        self.killCount = 0

    def update(self,oldGrayFrame,newGrayFrame):
        savPoints = self.oldPoints
        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, newGrayFrame,self.oldPoints,None,**self.lkParams)
        self.oldPoints = newPoints

        # if the point we're tracking hasn't moved, then the tracker isn't performing a useful service
        # so increase its kill count
        if pointDistance(savPoints,newPoints) == 0:
            self.kill()

        return newPoints, status, error

    def alreadyTracking(self,point):
        distance = pointDistance(self.oldPoints,point)

        # if the point in the argument is close to the position of the tracker, then we consider that we are
        # already tracking this point
        return distance < shortDistance

    def kill(self):
        self.killCount += 1

########################################################################################################################

cap = cv2.VideoCapture(inputVideo)

# grab a frame to determine its size (needed for the output video)
_,frame = cap.read()
frame = cv2.resize(frame, None, fx=fsca, fy=fsca)
height, width, c = frame.shape

# selecting mp4 as the output format doesn't seem to work, so I opted for avi instead
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvid = cv2.VideoWriter(outputVideo,fourcc,fps,(width,height))

# KNN is used here, but MOG2 would also work just fine
subtractor = cv2.createBackgroundSubtractorKNN(history=subtractorHist,dist2Threshold=subtractorThresh)

oldGray = np.array([[]])
trackerList = []

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=fsca, fy=fsca)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameMask = subtractor.apply(grayFrame)

    # we apply background subtraction to eliminate features that don't change from one frame to the next
    # such as elements of the HUD
    maskedFrame = cv2.bitwise_and(grayFrame, grayFrame, mask=frameMask)

    if len(oldGray) > 0:
        newTrackerList = []
        for tracker in trackerList:
            xSave,ySave = tracker.oldPoints.ravel()
            newPoints, status, error = tracker.update(oldGray,grayFrame)
            x, y = newPoints.ravel()
            # draw a green disk for the tracker
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # draw an arrow showing the direction of movement of the tracker
            xTip = int(xSave+(x-xSave)*trackerArrowScale)
            yTip = int(ySave+(y-ySave)*trackerArrowScale)
            cv2.arrowedLine(frame,(xSave,ySave),(xTip,yTip),(0,255,0),2,tipLength=0.5)

            # each time a tracker fails (status != 0) we increase its kill count
            if status.ravel() != 0 and error.ravel() > maxTrackerError:
                tracker.kill()
            # check that the tracker doesn't have too many kill counts
            if tracker.killCount < maxTrackerKills:
                newTrackerList.append(tracker)

        trackerList = newTrackerList[:]

    # Apply the Shi-Tomasi algorithm to find corners
    corners = cv2.goodFeaturesToTrack(maskedFrame, maxCorners, 0.01, 10)

    if corners is not None:
        corners = np.int0(corners)
        ptl = []
        for i in corners:
            x, y = i.ravel()
            # draw a red disk for the corner
            cv2.circle(frame, (x, y), 3, (0,0,255), -1)
            ptl.append((x,y))

        # next we group the corners by clusters of neighboring points
        clusters = hcluster.fclusterdata(ptl,shortDistance,criterion="distance")
        groups = [np.where(clusters==c_id)[0] for c_id in np.unique(clusters)]

        # now we look for large clusters of corner points to set a tracker
        for group in groups:
            if len(group) >= minClusterSize and len(trackerList) < maxTrackers:
                # we build a list of the points in the target cluster
                ptl1 = []
                for c in group:
                    ptl1.append(ptl[c])
                ptl1 = np.array(ptl1,dtype=np.float32)

                # and we draw a bounding rectangle around the points in the cluster
                x,y,w,h = cv2.boundingRect(ptl1)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # we choose the centroid of the rectangle as the point to track
                point = np.array([[x+w//2, y+h//2]], dtype=np.float32)

                # now we check whether we are already tracking this point
                tracked = False
                for tracker in trackerList:
                    if tracker.alreadyTracking(point):
                        tracked = True
                if not tracked:
                    trackerList.append(LKTracker(point,trackerWinSize,trackerMaxLevel,trackerCriteria))
                    print("now tracking %d points"%(len(trackerList)))

    cv2.imshow("Frame", frame)
    #cv2.imshow("maskedFrame", maskedFrame)
    oldGray = grayFrame.copy()
    outvid.write(frame)

    # pressing the ESC key should terminate the program
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
