import cv2
import numpy as np
import HandTrackingModule as htm
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##########
wCam, hCam = 640, 480
MirrorCamera = True
smoothness = 5
##########

pTime = 0
vol, volBar = 0, 400
area = 0
colorVol = (255, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volPer = round(volume.GetMasterVolumeLevelScalar() * 100)

detector = htm.handDetector(maxHands=1, detectionCon=0.7)

while True:
    success, img = cap.read()
    if MirrorCamera:
        img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        # print(area)

        if 200 < area < 1000:
            length, img, lineInfo = detector.findDistance(4, 8, img, True, 10)
            volBar = np.interp(length, [30, 200], [400, 150])
            volPer = np.interp(length, [30, 200], [0, 100])
            volPer = smoothness * round(volPer / smoothness)

            fingers = detector.fingersUp()
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                print("Volume Set", volPer, "%")
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 8, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(
        img,
        f"Volume: {int(volPer)} %",
        (20, 450),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 0),
        3,
    )
    cv2.putText(
        img,
        f"Vol Set: {round(volume.GetMasterVolumeLevelScalar() * 100)}%",
        (400, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        colorVol,
        2,
    )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow("Hand Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
