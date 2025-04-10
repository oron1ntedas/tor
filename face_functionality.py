import cv2
import numpy as np
from pynput.mouse import Button, Controller
import ctypes
from HandTrakingModule import handDetector
import time

class HandController:
    def __init__(self):
        self.wCam, self.hCam = 1280, 720
        user32 = ctypes.windll.user32
        self.wScr, self.hScr = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.frameReduction = int(self.wCam * 0.16)
        self.smoothening = 7

        self.detector = handDetector(maxHands=1, detectionCon=0.6)
        self.mouse = Controller()
        self.pLocX, self.pLocY = 0, 0
        self.last_click_time = 0
        self.is_active = False
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.RMB_pressed = False
        self.LMB_pressed = False
        self.click_smoother = 0

    def is_functionality_active(self):
        return self.is_active

    def activate(self):
        self.is_active = True
        print("Функционал управления активирован")

    def deactivate(self):
        self.is_active = False
        print("Функционал управления деактивирован")

    def process_frame(self, frame):
        if not self.is_active:
            return frame

        frame = self.detector.findHands(frame, draw=False)
        lmList = self.detector.findPosition(frame, draw=False)

        if lmList:
            fingers = self.detector.fingersUp()
            x_index, y_index = lmList[8][1], lmList[8][2]

            # Проверяем условие для прокрутки (все пальцы кроме большого опущены)
            if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
                self._handle_scroll(fingers, x_index, y_index, frame)
            # Проверяем условие для движения курсора (только указательный палец поднят)
            elif fingers[1] == 1 and fingers[2] == 0:
                self._handle_cursor_movement(x_index, y_index, frame)
            # Проверяем условие для кликов (указательный и средний пальцы подняты)
            elif fingers[1] == 1 and fingers[2] == 1:
                self._handle_clicks(fingers, lmList, frame)

        self._show_fps(frame)
        return frame

    def _handle_cursor_movement(self, x, y, frame):
        x3 = np.interp(x, (self.frameReduction, self.wCam - self.frameReduction), (0, self.wScr))
        y3 = np.interp(y, (self.frameReduction, self.hCam - self.frameReduction), (0, self.hScr))
        
        cLocX = self.pLocX + (x3 - self.pLocX) / self.smoothening
        cLocY = self.pLocY + (y3 - self.pLocY) / self.smoothening
        
        self.mouse.position = (cLocX, cLocY)
        cv2.circle(frame, (x, y), 15, (0, 0, 255), cv2.FILLED)
        self.pLocX, self.pLocY = cLocX, cLocY
        
        if self.click_smoother >= 30:
            self.RMB_pressed, self.LMB_pressed = False, False
        self.click_smoother += 1

    def _handle_clicks(self, fingers, lmList, frame):
        lenghtIM, frame, _ = self.detector.findDistance(8, 12, frame)
        lenghtMR, frame, _ = self.detector.findDistance(12, 16, frame)
        
        if not self.RMB_pressed and fingers[3] == 1 and lenghtMR < 55:
            cv2.circle(frame, (lmList[12][1], lmList[12][2]), 15, (0, 255, 0), cv2.FILLED)
            self.mouse.click(Button.right, 1)
            self.RMB_pressed, self.LMB_pressed = True, False
        
        elif not self.LMB_pressed and lenghtIM < 55 and fingers[3] == 0:
            mid_point = ((lmList[8][1] + lmList[12][1])//2, (lmList[8][2] + lmList[12][2])//2)
            cv2.circle(frame, mid_point, 15, (0, 255, 0), cv2.FILLED)
            self.mouse.click(Button.left, 1)
            self.RMB_pressed, self.LMB_pressed = False, True
        
        self.click_smoother = 0

    def _handle_scroll(self, fingers, x, y, frame):
        # Проверяем, что указательный, средний и безымянный пальцы опущены
        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
            # Рисуем синий кружок в точке прокрутки
            cv2.circle(frame, (x, y), 10, (255, 0, 0), cv2.FILLED)
            # Если большой палец опущен - прокрутка вверх, иначе вниз
            self.mouse.scroll(0, 2 if fingers[0] == 0 else -2)

    def _show_fps(self, frame):
        self.fps_counter += 1
        now = time.time()
        if now - self.last_fps_time >= 1:
            fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = now
            cv2.putText(frame, f'FPS: {fps}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
