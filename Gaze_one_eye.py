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

    # horLineLeft = cv2.line(frame, leftPoint, rightPoint, (255, 0, 0), 1)
    # verLineLeft = cv2.line(frame, topPoint, bottomPoint, (255, 0, 0), 1)

    ver_line_lenght = hypot((topPoint[0] - bottomPoint[0]), (topPoint[1] - bottomPoint[1]))
    return ver_line_lenght

counter = 1

while True:
    _, frame = cap.read()
    
    fire_name = f"Picture{counter}.png"
    fire = cv2.imread(fire_name)
    if counter == 4: counter = 1
    else: counter += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)

        # # Blinking Detection:
        # leftEye = get_blinking([36, 37, 38, 39, 40, 41], landmarks)
        # rightEye = get_blinking([42, 43, 44, 45, 46, 47], landmarks)
        # 
        # if leftEye < 9 and rightEye < 9:
        #     cv2.putText(frame, "Blinking", (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (0, 0, 0), 2)
            
            
        # Gaze Detection:
        rightEyePoints = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                 (landmarks.part(37).x, landmarks.part(37).y),
                                 (landmarks.part(38).x, landmarks.part(38).y),
                                 (landmarks.part(39).x, landmarks.part(39).y),
                                 (landmarks.part(40).x, landmarks.part(40).y),
                                 (landmarks.part(41).x, landmarks.part(41).y) ], np.int32)
    
        Point36, Point39 = rightEyePoints[0], rightEyePoints[3]
        hor_len = int(hypot((Point39[0] - Point36[0]), (Point39[1] - Point36[1])) / 2)
        
        # rightEyePoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)], np.int32)
        # cv2.polylines(frame, [poly_one_eye], True, (0, 0, 255), 1)
        
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        
        cv2.polylines(mask, [rightEyePoints], True, 255, 1)
        cv2.fillPoly(mask, [rightEyePoints], 255)
        
        dim_fire = (120, 100)
        resized_fire = cv2.resize(fire, dim_fire, interpolation = cv2.INTER_AREA)
        
        point_mid = [int( ((rightEyePoints[2][0]-rightEyePoints[1][0])/2)-(hor_len) ), 
                     int( (rightEyePoints[2][1]-rightEyePoints[1][1])/2 )]
        h_fire, w_fire, _ = resized_fire.shape
        
        
        Object = cv2.bitwise_and(frame, frame, mask=mask)
        
        
        # output[point_mid[0]+h_fire: point_mid[0], point_mid[1]: point_mid[1]+w_fire ] = resized_fire
        
        rightEyePoints[0][0] -= hor_len
        rightEyePoints[3][0] += hor_len
        rightEyePoints[4][0] += (hor_len/2)
        rightEyePoints[4][1] += hor_len
        rightEyePoints[5][0] -= (hor_len/2)
        rightEyePoints[5][1] += hor_len
        
        devilPionts = np.array([
                       rightEyePoints[0], # 0
                       rightEyePoints[5], # 1
                       rightEyePoints[4], # 2
                       rightEyePoints[3], # 3
                       rightEyePoints[4], # 4
                       rightEyePoints[4], # 5
                       rightEyePoints[5], # 6
                       rightEyePoints[5]  # 7
                       ], np.int32)
        
        devilPionts[5][0] += (hor_len/2.7)*10
        devilPionts[5][1] += hor_len*30
        devilPionts[6][0] -= (hor_len/2.7)*10
        devilPionts[6][1] += hor_len*30
        
        cv2.polylines(Object, [devilPionts], True, (0, 0, 180), 3)
        cv2.fillPoly(Object, [devilPionts], (0, 0, 100))
        
        background = np.ones((height, width, 3), np.uint8)
        # cv2.polylines(mask, [devilPionts], True, 0, 1)
        # cv2.fillPoly(mask, [devilPionts], 0)
        
        # output = cv2.bitwise_and(background, background, mask=Object)
        
        
        # cv2.imshow("Eye", thresholdEye)
        cv2.imshow("Mask Eyes", Object)
        # cv2.imshow("Mask Background", background)

    # cv2.imshow("frame", frame)
    cv2.imshow("fire", resized_fire)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()