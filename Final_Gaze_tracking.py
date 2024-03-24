import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def mid_point(p1, p2):
    return int((p1.x + p2.x) / 2 ), int((p1.y + p2.y) / 2)


def get_blinking(eyePoints, face):
    leftPoint = (face.part(eyePoints[0]).x, face.part(eyePoints[0]).y)
    rightPoint = (face.part(eyePoints[3]).x, face.part(eyePoints[3]).y)

    topPoint = mid_point(face.part(eyePoints[1]), face.part(eyePoints[2]))
    bottomPoint = mid_point(face.part(eyePoints[4]), face.part(eyePoints[5]))

    horLineLeft = cv2.line(frame, leftPoint, rightPoint, (255, 0, 0), 1)
    verLineLeft = cv2.line(frame, topPoint, bottomPoint, (255, 0, 0), 1)

    ver_line_lenght = hypot((topPoint[0] - bottomPoint[0]), (topPoint[1] - bottomPoint[1]))
    return ver_line_lenght

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)

        # Blinking Detection:
        leftEye = get_blinking([36, 37, 38, 39, 40, 41], landmarks)
        rightEye = get_blinking([42, 43, 44, 45, 46, 47], landmarks)
        
        if leftEye < 9 and rightEye < 9:
            cv2.putText(frame, "Blinking", (240, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (0, 0, 0), 2)
            
        # Gaze Detection:
        leftEyePoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)], np.int32)
        rightEyePoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42,48)], np.int32)
        
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        
        cv2.polylines(mask, [leftEyePoints], True, 255, 1)
        cv2.fillPoly(mask, [leftEyePoints], 255)
        cv2.polylines(mask, [rightEyePoints], True, 255, 1)
        cv2.fillPoly(mask, [rightEyePoints], 255)
        
        mask_Eye = cv2.bitwise_and(gray, gray, mask=mask)
        cv2.imshow("Mask_Eyes", mask_Eye)

        
        min_x, max_x = np.min(leftEyePoints[:, 0]), np.max(leftEyePoints[:, 0])
        min_y, max_y = np.min(leftEyePoints[:, 1]), np.max(leftEyePoints[:, 1])
        just_Left_Eye = mask_Eye[min_y: max_y, min_x: max_x]
        
        min_x, max_x = np.min(rightEyePoints[:, 0]), np.max(rightEyePoints[:, 0])
        min_y, max_y = np.min(rightEyePoints[:, 1]), np.max(rightEyePoints[:, 1])
        just_Right_Eye = mask_Eye[min_y: max_y, min_x: max_x]
        
        _, thresholdLeftEye = cv2.threshold(just_Left_Eye, 60, 255, cv2.THRESH_BINARY)
        _, thresholdRightEye = cv2.threshold(just_Right_Eye, 60, 255, cv2.THRESH_BINARY)
        
        height, width = thresholdLeftEye.shape
        left_side_thresh = thresholdLeftEye[0:height, 0:int(width/2)]
        right_side_thresh = thresholdLeftEye[0:height, int(width/2):width]
        left_eye_nums = [cv2.countNonZero(right_side_thresh), cv2.countNonZero(left_side_thresh)]
        
        height, width = thresholdRightEye.shape
        left_side_thresh = thresholdRightEye[0:height, 0:int(width/2)]
        right_side_thresh = thresholdRightEye[0:height, int(width/2):width]
        right_eye_nums = [cv2.countNonZero(right_side_thresh), cv2.countNonZero(left_side_thresh)]
        
        cv2.putText(frame, str(left_eye_nums), (10,130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (0, 0, 0), 2)
        cv2.putText(frame, str(right_eye_nums), (150,130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (0, 0, 0), 2)
        
        left_eye_gaze = int(left_eye_nums[0] - left_eye_nums[1])
        right_eye_gaze = int(right_eye_nums[0] - right_eye_nums[1])
        if left_eye_gaze > 20 and right_eye_gaze > 20:
            cv2.putText(frame, "Left", (250,250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
        elif left_eye_gaze < -20 and right_eye_gaze < -20:
            cv2.putText(frame, "Right", (250,250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Center", (250,250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
        
        
        min_height = min(thresholdLeftEye.shape[0], thresholdRightEye.shape[0])
        min_width = min(thresholdLeftEye.shape[1], thresholdRightEye.shape[1])
        img1 = thresholdLeftEye[0:min_height, 0:min_width]
        img2 = thresholdRightEye[0:min_height, 0:min_width]
        
        img1 = cv2.resize(img1, None, fx=5, fy=5)
        img2 = cv2.resize(img2, None, fx=5, fy=5)
        full_thresh_eye = np.concatenate((img1, img2), axis=1)
        cv2.imshow("Thresh_Eyes", full_thresh_eye)
        
        

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
