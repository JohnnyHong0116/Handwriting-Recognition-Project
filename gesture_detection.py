import numpy as np
from hand_model import mp_hands

def detect_hand_gesture(landmarks, w, h):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    distances = [
        np.linalg.norm(np.array([thumb_tip.x * w, thumb_tip.y * h]) - np.array([wrist.x * w, wrist.y * h])),
        np.linalg.norm(np.array([index_tip.x * w, index_tip.y * h]) - np.array([wrist.x * w, wrist.y * h])),
        np.linalg.norm(np.array([middle_tip.x * w, middle_tip.y * h]) - np.array([wrist.x * w, wrist.y * h])),
        np.linalg.norm(np.array([ring_tip.x * w, ring_tip.y * h]) - np.array([wrist.x * w, wrist.y * h])),
        np.linalg.norm(np.array([pinky_tip.x * w, pinky_tip.y * h]) - np.array([wrist.x * w, wrist.y * h]))
    ]

    if all(dist > 0.3 * w for dist in distances):
        return 'open'
    elif all(dist < 0.15 * w for dist in distances):
        return 'fist'
    return 'other'

def detect_ok_gesture(landmarks, w, h):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]

    thumb_index_distance = np.linalg.norm(np.array([thumb_tip.x * w, thumb_tip.y * h]) - np.array([index_tip.x * w, index_tip.y * h]))
    thumb_index_dip_distance = np.linalg.norm(np.array([thumb_tip.x * w, thumb_tip.y * h]) - np.array([index_dip.x * w, index_dip.y * h]))

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    middle_tip_distance = np.linalg.norm(np.array([middle_tip.x * w, middle_tip.y * h]) - np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h]))
    ring_tip_distance = np.linalg.norm(np.array([ring_tip.x * w, ring_tip.y * h]) - np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x * w, landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y * h]))
    pinky_tip_distance = np.linalg.norm(np.array([pinky_tip.x * w, pinky_tip.y * h]) - np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP].x * w, landmarks[mp_hands.HandLandmark.PINKY_MCP].y * h]))

    if (thumb_index_distance < 20 and thumb_index_dip_distance < 20 and
            middle_tip_distance > 50 and ring_tip_distance > 50 and pinky_tip_distance > 50):
        return True
    return False

def calculate_circle_radius(landmarks, w, h):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.linalg.norm(np.array([thumb_tip.x * w, thumb_tip.y * h]) - np.array([index_tip.x * w, index_tip.y * h]))
    radius = int(distance / 2)
    return max(5, min(50, radius))

def detect_pinch_gesture(landmarks, w, h):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate the distance between thumb and index finger tip
    thumb_index_distance = np.linalg.norm(np.array([thumb_tip.x * w, thumb_tip.y * h]) - np.array([index_tip.x * w, index_tip.y * h]))

    # Check if thumb and index finger are touching
    thumb_index_pinch = thumb_index_distance < 20

    # Ensure other fingers are extended (not curled into a fist)
    middle_extended = middle_tip.y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_extended = ring_tip.y < landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_extended = pinky_tip.y < landmarks[mp_hands.HandLandmark.PINKY_MCP].y

    if thumb_index_pinch and middle_extended and ring_extended and pinky_extended:
        return True, (int(thumb_tip.x * w), int(thumb_tip.y * h)), (int(index_tip.x * w), int(index_tip.y * h))
    
    return False, (int(thumb_tip.x * w), int(thumb_tip.y * h)), (int(index_tip.x * w), int(index_tip.y * h))


def is_fist_gesture(hand_landmarks):
    # Extract the relevant landmarks
    index_finger_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger_tip = hand_landmarks[mp_hands.HandLandmark.PINKY_TIP]

    index_finger_mcp = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_finger_mcp = hand_landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Check if the fingertips are close to the MCP joints (fingers curled)
    def is_finger_curled(mcp, tip):
        return tip.y > mcp.y  # Fingertip below the MCP joint in y-axis (hand is flipped, y increases downwards)

    return (
        is_finger_curled(index_finger_mcp, index_finger_tip) and
        is_finger_curled(middle_finger_mcp, middle_finger_tip) and
        is_finger_curled(ring_finger_mcp, ring_finger_tip) and
        is_finger_curled(pinky_finger_mcp, pinky_finger_tip)
    )
