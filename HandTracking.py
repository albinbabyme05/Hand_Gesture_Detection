from cv2 import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2,model_complexity =1, min_detection_accuracy=0.5, min_tracking_accuracy=0.5):
        """Initialising the Hands function"""
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_accuracy = min_detection_accuracy
        self.min_tracking_accuracy = min_tracking_accuracy

        self.mp_Hands = mp.solutions.hands
        self.hands = self.mp_Hands.Hands(self.mode, self.max_hands, self.model_complexity, self.min_detection_accuracy,
                                         self.min_tracking_accuracy)
        self.mpDraw = mp.solutions.drawing_utils

    def capture_Hand_frame(self, frame, draw=True):
        """Uses captured frames to draw landmarks"""
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_RGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLandmarks,
                                               self.mp_Hands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):

        my_list = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for ID, land_mark_ID in enumerate(myHand.landmark):
                height, width, channel = frame.shape
                cx, cy = int(land_mark_ID.x * width), int(land_mark_ID.y * height)
                my_list.append([ID, cx, cy])
                # if draw:
                #     cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return my_list


def detectHands():
    """Uses webcam to capture the video frames"""
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = detector.capture_Hand_frame(frame)
        my_list = detector.findPosition(frame)
        if len(my_list) != 0:
            print(my_list[4])
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        cv2.imshow("video", frame)
        cv2.waitKey(1)
        if cv2.waitKey(10) % 256 == 27:
            print('Escape pressed. Closing the application')
            break


if __name__ == "__main__":
    detectHands()
