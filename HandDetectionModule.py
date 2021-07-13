# some imports
import cv2
import mediapipe as mp
import time


# object for hand detection and the parameters
# 1. static_image_mode=False --> If this is false it will sometimes
# detect the hand and other times track it based on the confidence, that means, if it has confidence that the hand
# has been detected it will start to track and if it loses the hand then it will again detect, but if it is true,
# then it will always detect and never track.
#
# 2. max_num_hands=2, --> The number of hands to detect and track at one
# time in the frame
#
# 3. min_detection_confidence=0.5, --> Confidence in detecting the hands
#
# 4. min_tracking_confidence=0.5) --> Confidence in being able to track the hands, if it goes below 50% then it will
# agian start the detection

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon

        # need to do this before we initiate the model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        # to draw on the hand
        self.mpDraw = mp.solutions.drawing_utils

    def detect_hands(self, img, draw=True):
        # hands only uses rgb images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)


        if self.results.multi_hand_landmarks:
            # hands are detected
            for handlms in self.results.multi_hand_landmarks:  # for all landmarks on all hands
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

        return img

    # function to find all the positions of the landmarks on the hand
    def find_position(self, img, handNo=0, draw=True):

        lmlist = []  # list for storing all the positions

        # if we detect hand/hands
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  # get only one hand as mentioned
            # for the hand detected
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist


def main():
    pTime = 0
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    while True:
        ret, img = cap.read()
        if ret:
            img = detector.detect_hands(img)
            lmlist = detector.find_position(img)
            if len(lmlist) != 0:
                print(lmlist[4])
            # for fps
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # to display it on screen
            cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            # show the image
            cv2.imshow("image", img)
            cv2.waitKey(1)

        else:
            print("no image from camera")


if __name__ == "__main__":
    main()
