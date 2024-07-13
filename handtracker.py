import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

def is_fist(landmarks):
    wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
    thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]

    def distance(point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    d_thumb = distance(wrist, thumb_tip)
    d_index = distance(wrist, index_tip)
    d_middle = distance(wrist, middle_tip)
    d_ring = distance(wrist, ring_tip)
    d_pinky = distance(wrist, pinky_tip)

    if d_thumb < 0.2 and d_index < 0.2 and d_middle < 0.2 and d_ring < 0.2 and d_pinky < 0.2:
        return True
    return False

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
                center = (int(image.shape[1] * wrist.x), int(image.shape[0] * wrist.y))
                cv2.ellipse(image, center, (50, 50), -90, 0, dial_angle, (255, 0, 0), 5)
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