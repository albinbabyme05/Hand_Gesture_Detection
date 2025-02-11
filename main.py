from cv2 import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands()
mpDraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands         .process(frame_RGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for ID, land_mark_ID in enumerate(handLandmarks.landmark):
                print(f"The id no is : {ID},"
                      f"The Land mark is : {land_mark_ID}")
                height, width, channel = frame.shape
                cx, cy = int(land_mark_ID.x*width), int(land_mark_ID.y*height)
                print(f"The id value is : {ID},\n"
                      f"The center value x is :{cx},\n"
                      f"The center value y is :{cy}\n \n"
                      )
            mpDraw.draw_landmarks(frame, handLandmarks, mp_Hands.HAND_CONNECTIONS)

        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time

        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

    cv2.imshow("video", frame)
    if cv2.waitKey(10) % 256 == 27:
        print('Escape pressed. Closing the application')
        break

    cv2.waitKey(1)