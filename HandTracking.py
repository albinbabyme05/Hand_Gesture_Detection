import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, max_hands=2, min_detection_accuracy=0.5, min_tracking_accuracy=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.min_detection_accuracy = min_detection_accuracy
        self.min_tracking_accuracy = min_tracking_accuracy

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, min_detection_accuracy, min_tracking_accuracy)
        self.mp_draw = mp.solutions.drawing_utils

    def capture_Hand_frame(self, frame):
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_RGB)
        return frame
