import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:
    while True:
        ret, frame = camera.read()

        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("CAM", frame)

        if cv2.waitKey(10) == 27:
            break

camera.release()
cv2.destroyAllWindows()
