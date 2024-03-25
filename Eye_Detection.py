import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
        
        leftEyePoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)], np.int32)
        rightEyePoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42,48)], np.int32)
        
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        
        cv2.polylines(mask, [leftEyePoints], True, 255, 1)
        cv2.fillPoly(mask, [leftEyePoints], 255)
        cv2.polylines(mask, [rightEyePoints], True, 255, 1)
        cv2.fillPoly(mask, [rightEyePoints], 255)
        
        mask_Eye = cv2.bitwise_and(gray, gray, mask=mask)
        
        cv2.imshow("Mask Eyes", mask_Eye)

        
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
