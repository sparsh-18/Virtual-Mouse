import cv2
import math
import mediapipe as mp
import autopy
import numpy as np


############################################################

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# print(volume.GetMasterVolumeLevel(), volume.GetVolumeRange())

#############################################################

# size of screen
wScr, hScr = autopy.screen.size()

# to smoothen the values of mouse coordinates so that it wont flicker
smoother = 7

# previous mouse coordinates
prevx = prevy = 0

# rectangle frame in which the mouse works
frameR = 100

# webcam frame
frameWidth = 640
frameHeight = 480

# video capture
cap1 = cv2.VideoCapture(0)
cap1.set(3, frameWidth)
cap1.set(4, frameHeight)

# mediapipe library
mphands = mp.solutions.hands

hands = mphands.Hands()

# a list having index values of tip and pip of index, middle, ring, pinky
fingerList = [
    [mphands.HandLandmark.INDEX_FINGER_TIP, mphands.HandLandmark.INDEX_FINGER_PIP],
    [mphands.HandLandmark.MIDDLE_FINGER_TIP, mphands.HandLandmark.MIDDLE_FINGER_PIP],
    [mphands.HandLandmark.RING_FINGER_TIP, mphands.HandLandmark.RING_FINGER_PIP],
    [mphands.HandLandmark.PINKY_TIP, mphands.HandLandmark.PINKY_PIP],
    [mphands.HandLandmark.INDEX_FINGER_PIP, mphands.HandLandmark.THUMB_TIP]
]

# to draw traced hand
mpdraw = mp.solutions.drawing_utils

while True:
    success, img = cap1.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(imgRGB)

    # index, middle, ring, pinky, thumb    1-up  0-down
    fingers = [0, 0, 0, 0, 0]

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

            # updating list with fingers up
            for i in range(0, 4):
                if handLms.landmark[fingerList[i][0]].y < handLms.landmark[fingerList[i][1]].y:
                    fingers[i] = 1

            if handLms.landmark[fingerList[4][0]].x < handLms.landmark[fingerList[4][1]].x:
                fingers[4] = 1

    # creating rectangle in which mouse works
    cv2.rectangle(img, (frameR, frameR), (frameWidth - frameR, frameHeight - frameR), (255, 0, 255), 1)

    # moving mouse when only index up
    if fingers[0] == 1 and fingers[1] == 0 and fingers[4] == 0:
        # x and y coordinates of index finger
        index_tip_x = int(handLms.landmark[fingerList[0][0]].x * frameWidth)
        index_tip_y = int(handLms.landmark[fingerList[0][0]].y * frameHeight)
        # draw circle around the tip of index finger
        cv2.circle(img, (index_tip_x, index_tip_y), 23, (255, 0, 255), 20)

        x3 = np.interp(index_tip_x, (frameR, frameWidth - frameR), (0, wScr))
        y3 = np.interp(index_tip_y, (frameR, frameHeight - frameR), (0, hScr))

        # smoothen the values
        smooth_x = prevx + (x3 - prevx) / smoother
        smooth_y = prevy + (y3 - prevy) / smoother

        prevx = smooth_x
        prevy = smooth_y

        # flipping x so mouse must go left and right when finger moves left and right respectively
        newx = wScr - smooth_x

        if newx > 0 and smooth_y > 0:
            autopy.mouse.move(newx, smooth_y)

    # both index and middle up for clicking
    if fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 0:
        index_tip_x = int(handLms.landmark[fingerList[0][0]].x * frameWidth)
        index_tip_y = int(handLms.landmark[fingerList[0][0]].y * frameHeight)

        middle_tip_x = int(handLms.landmark[fingerList[1][0]].x * frameWidth)
        middle_tip_y = int(handLms.landmark[fingerList[1][0]].y * frameHeight)

        # if distance between index and tip is less than 30 then click
        if index_tip_x - middle_tip_x < 30 and middle_tip_y < index_tip_y:
            autopy.mouse.click()

    # index and thumb up
    if fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:

        index_tip_x = int(handLms.landmark[fingerList[0][0]].x * frameWidth)
        index_tip_y = int(handLms.landmark[fingerList[0][0]].y * frameHeight)

        thumb_tip_x = int(handLms.landmark[mphands.HandLandmark.THUMB_TIP].x * frameWidth)
        thumb_tip_y = int(handLms.landmark[mphands.HandLandmark.THUMB_TIP].y * frameHeight)

        center_x = int((index_tip_x + thumb_tip_x) / 2)
        center_y = int((index_tip_y + thumb_tip_y) / 2)

        cv2.line(img, (index_tip_x, index_tip_y), (thumb_tip_x, thumb_tip_y), (255, 255, 0), thickness=4)

        cv2.circle(img, (center_x, center_y), 10, (255, 0, 255), cv2.FILLED)

        # ditance between thumb and index
        dist = math.sqrt((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2)

        # from get vol range we know that -65.25 is vol 0% and 0 is 100% vol
        # interp changes the range from 20,200 (we know from printing distance) to -65,0
        vol = np.interp(dist, [20, 200], [-65.25, 0])

        volume.SetMasterVolumeLevel(vol, None)


    # showing webcam q to exit
    cv2.imshow("title", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
