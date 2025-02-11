import os
import time
from cv2 import cv2
import HandTracking as ht


cam_width = 640
cam_height = 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

path = "Finger Data"
list_dir = os.listdir(path)

superimpose_list = []
for image in list_dir:
    retr_image = cv2.imread(f"{path}/{image}")
    superimpose_list.append(retr_image)

# print(len(superimpose_list))
start_time = 0

detector = ht.HandDetector(min_detection_accuracy=0.80)
fingertip_id = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    frame = detector.capture_Hand_frame(frame)
    my_list = detector.findPosition(frame, draw=False)
    #print(my_list)
    if len(my_list) != 0:
        fingers = []

        if my_list[fingertip_id[0]][1] > my_list[fingertip_id[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if my_list[fingertip_id[id]][2] < my_list[fingertip_id[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        total_Fingers = fingers.count(1)
        if total_Fingers == 1:
            print("Index Finger open")
            cv2.putText(frame, "Index Finger", (100, 40), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 0, 0), 2)
        elif total_Fingers == 2:
            cv2.putText(frame, "Index and Middle Finger", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            print("Index and Middle finger open")
        elif total_Fingers == 3:
            cv2.putText(frame, "Index, Middle and Ring Finger", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            print("Index, Middle and Ring Finger open")
        elif total_Fingers == 4:
            cv2.putText(frame, "Index, Middle, Ring and Pinky finger", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            print("Index, Middle, Ring and Pinky finger open")
        elif total_Fingers == 5:
            cv2.putText(frame, "All fingers", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            print("All fingers are open")
        elif total_Fingers == 0:
            cv2.putText(frame, "Fist", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            print("Fist")

        # print(total_Fingers)
        # height, width, channel = superimpose_list[total_Fingers-1].shqape
        # frame[0:height, 0:width] = superimpose_list[total_Fingers-1]

    current_time = time.time()
    fps = 1 / (current_time - start_time)
    start_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (550, 40), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 0), 2)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
    key = cv2.waitKey(1) % 256
    if key == 27:
        print("Escape pressed. Closing the application")
        break
