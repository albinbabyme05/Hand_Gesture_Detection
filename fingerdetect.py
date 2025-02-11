import cv2
import HandTracking as ht

cap = cv2.VideoCapture(0)
detector = ht.HandDetector(min_detection_accuracy=0.8)
fingertip_id = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    frame = detector.capture_Hand_frame(frame)
    my_list = detector.findPosition(frame, draw=False)

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

        total_fingers = fingers.count(1)
        cv2.putText(frame, f"Fingers: {total_fingers}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
