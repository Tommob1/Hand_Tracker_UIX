import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

def is_fist(landmarks):
    fingers_folded = 0
    fingers = [
        (mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.THUMB_IP, mp.solutions.hands.HandLandmark.THUMB_MCP),
        (mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP),
        (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_PIP, mp.solutions.hands.HandLandmark.RING_FINGER_MCP),
        (mp.solutions.hands.HandLandmark.PINKY_TIP, mp.solutions.hands.HandLandmark.PINKY_PIP, mp.solutions.hands.HandLandmark.PINKY_MCP),
    ]
    
    for tip_id, pip_id, mcp_id in fingers:
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        mcp = landmarks[mcp_id]
        if tip.y > pip.y and pip.y > mcp.y:
            fingers_folded += 1
    
    thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
    
    if thumb_tip.x < wrist.x and thumb_tip.y > index_mcp.y:
        fingers_folded += 1
    
    if fingers_folded == 5:
        return True
    else:
        return False

def get_hand_center(landmarks):
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    center_x = sum(x_coords) / len(landmarks)
    center_y = sum(y_coords) / len(landmarks)
    return center_x, center_y

cap = cv2.VideoCapture(0)
hands = mphands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

dial_angle = 0
baseline_angle = None
tracking_hand = None
tracking_hand_landmarks = None

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks_list = results.multi_hand_landmarks

        if len(hand_landmarks_list) == 2:
            hand_1 = hand_landmarks_list[0].landmark
            hand_2 = hand_landmarks_list[1].landmark

            if is_fist(hand_1):
                tracking_hand = hand_2
                tracking_hand_landmarks = hand_landmarks_list[1]
            elif is_fist(hand_2):
                tracking_hand = hand_1
                tracking_hand_landmarks = hand_landmarks_list[0]
            else:
                tracking_hand = None
                baseline_angle = None

            if tracking_hand:
                thumb_tip = tracking_hand[mp.solutions.hands.HandLandmark.THUMB_TIP]
                wrist = tracking_hand[mp.solutions.hands.HandLandmark.WRIST]

                dx = thumb_tip.x - wrist.x
                dy = thumb_tip.y - wrist.y
                current_angle = np.arctan2(dy, dx) * 180 / np.pi
                print(f'Current angle: {current_angle}')

                if baseline_angle is None:
                    baseline_angle = current_angle
                    print(f'Baseline angle set to: {baseline_angle}')

                dial_angle = current_angle - baseline_angle
                dial_angle = (dial_angle + 180) % 360 - 180

                center_x, center_y = get_hand_center(tracking_hand)
                center = (int(image.shape[1] * center_x), int(image.shape[0] * center_y))
                
                cv2.circle(image, center, 200, (255, 0, 0), 3)
                cv2.ellipse(image, center, (200, 200), -90, 0, dial_angle, (0, 255, 0), 5)
                
                for i in range(0, 360, 10):
                    angle_rad = np.deg2rad(i)
                    pt1 = (int(center[0] + 200 * np.cos(angle_rad)), int(center[1] + 200 * np.sin(angle_rad)))
                    pt2 = (int(center[0] + 220 * np.cos(angle_rad)), int(center[1] + 220 * np.sin(angle_rad)))
                    cv2.line(image, pt1, pt2, (255, 0, 0), 3)
                    if i % 30 == 0:
                        pt3 = (int(center[0] + 250 * np.cos(angle_rad)), int(center[1] + 250 * np.sin(angle_rad)))
                        cv2.putText(image, str(i), pt3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.putText(image, f'{int(dial_angle)}', (center[0], center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
                print(f'Dial angle: {dial_angle}')

        for i, hand_landmarks in enumerate(hand_landmarks_list):
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mphands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=5),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5))

    cv2.imshow('Hand Tracker', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
