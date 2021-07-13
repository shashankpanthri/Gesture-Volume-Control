# some imports
import cv2
import time
import HandDetectionModule as hdm
import numpy as np
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
widthCam, heightCam = 640, 480
################################

pTime = 0

cap = cv2.VideoCapture(1)
# to make the screen bigger we set the width and height parameters to above set values
cap.set(3, widthCam)
cap.set(4, heightCam)

# the detection of the hand is sometimes jittery i.e. it detects extra hand due to low confidence preset, so
# we will change that
detector = hdm.HandDetector(detectCon=0.72, maxHands=1)

# initializations
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# min and max volume
minVol = volRange[0]
maxVol = volRange[1]
vol2 = 0
lengthofpinkyandring = 0
volbarlength = 400
volbarper = 0

while True:
    ret, img = cap.read()
    if ret:
        img = detector.detect_hands(img)
        lmlist = detector.find_position(img, draw=False)
        # now from this list of landmarks according to the picture of hand lanmarks and their numbers we need:
        # 1. tip of thumb - 4
        # 2. tip of index finger - 8
        if len(lmlist) != 0:
            # print(lmlist[4], lmlist[8])

            # points
            x1, y1 = lmlist[4][1], lmlist[4][2]  # thumb
            x2, y2 = lmlist[8][1], lmlist[8][2]  # index
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center of the line
            # for the stop gesture
            x3, y3 = lmlist[20][1], lmlist[20][2]  # tip of pinky
            x4, y4 = lmlist[15][1], lmlist[15][2]  # second to top of ring

            # to make it easier, we can create a circle around them
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            # a line between them
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # for the length of this line, we use hypot function
            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)
            # length of the stop gesture
            lengthofpinkyandring = math.hypot(x4 - x3, y4 - y3)
            # print("pinky ring", lengthofpinkyandring)

            # hand range --> 30 - 260
            # volume range --> -65 - 0
            # for conversion of above range to lower one
            vol1 = np.interp(length, [1, 260], [1, 100])
            vol2 = np.interp(vol1, [1, 100], [minVol, maxVol])
            # print(int(length),vol)
            if lengthofpinkyandring > 70:
                volume.SetMasterVolumeLevel(vol2, None)

            else:
                cv2.circle(img, (x3, y3), 20, (255, 0, 0), cv2.FILLED)

            if length < 20:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        if lengthofpinkyandring > 70:
            volbarlength = np.interp(vol2, [minVol, maxVol], [400, 150])
            volbarper = np.interp(vol2, [minVol, maxVol], [0,100])
        cv2.rectangle(img, (40, 150), (75, 400),  (255, 0, 0), 2)
        cv2.rectangle(img, (40, int(volbarlength)), (75, 400),  (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(volbarper)), (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # for fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # to display it on screen
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # show the image
        cv2.imshow("image", img)
        cv2.waitKey(1)

    else:
        print("no image from camera")
