import cv2
import mediapipe as mp
import math

class handDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.hands = mp.solutions.hands.Hands(static_image_mode=mode,
                                              max_num_hands=maxHands,
                                              model_complexity=modelComplexity,
                                              min_detection_confidence=detectionCon,
                                              min_tracking_confidence=trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if not self.lmList:
            return fingers
        # Проверка большого пальца
        fingers.append(1 if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1] else 0)
        # Остальные пальцы
        for i in range(1, 5):
            fingers.append(1 if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2] else 0)
        return fingers

    def findDistance(self, p1, p2, img=None, draw=True, r=8):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        if draw and img is not None:
            color = (0, 255, 0) if length < 55 else (0, 0, 255)
            for pt in [(x1, y1), (x2, y2), (cx, cy)]:
                cv2.circle(img, pt, r, color, cv2.FILLED)
        return length, img, [x1, y1, x2, y2, cx, cy]
