import cv2
import numpy as np
from hand_model import mp_hands
from geometry_utils import calculate_overlap_area

def draw_labeled_bounding_boxes(frame, joints, w, h, distance, calibration_complete):
    base_tip_box_size = 12
    
    if distance > 0:
        adjusted_tip_box_size = int(base_tip_box_size / (distance**0.5))
    else:
        adjusted_tip_box_size = base_tip_box_size
    
    adjusted_tip_box_size = max(2, min(15, adjusted_tip_box_size))

    overlap_color = (0, 255, 255)
    boxes = []

    for label, joint in joints.items():
        joint_x = int(joint.x * w)
        joint_y = int(joint.y * h)
        top_left = (joint_x - adjusted_tip_box_size, joint_y - adjusted_tip_box_size)
        bottom_right = (joint_x + adjusted_tip_box_size, joint_y + adjusted_tip_box_size)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(frame, label, (joint_x - adjusted_tip_box_size, joint_y - adjusted_tip_box_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        boxes.append((top_left, bottom_right))

    if calibration_complete:
        total_overlap_area = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box1 = boxes[i]
                box2 = boxes[j]
                overlap_area = calculate_overlap_area(box1, box2)
                total_overlap_area += overlap_area
                if overlap_area > 0:
                    overlap_top_left = (max(box1[0][0], box2[0][0]), max(box1[0][1], box2[0][1]))
                    overlap_bottom_right = (min(box1[1][0], box2[1][0]), min(box1[1][1], box2[1][1]))
                    cv2.rectangle(frame, overlap_top_left, overlap_bottom_right, overlap_color, -1)
        return total_overlap_area
    return 0

def draw_trapezoid(frame, top_left, top_right, bottom_left, bottom_right):
    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(255, 255, 0), thickness=2)

def draw_finger_rectangle(frame, landmarks, w, h, calibration_complete):
    if not calibration_complete:
        return False

    # Get the positions of the finger tips and MCP joints
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

    # MCP joints for index, middle, and ring fingers
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    # Calculate distances to check if the hand is wide open and fingers are straight
    tip_distances = [
        np.linalg.norm(np.array([index_tip.x * w, index_tip.y * h]) - np.array([middle_tip.x * w, middle_tip.y * h])),
        np.linalg.norm(np.array([middle_tip.x * w, middle_tip.y * h]) - np.array([ring_tip.x * w, ring_tip.y * h])),
        np.linalg.norm(np.array([ring_tip.x * w, ring_tip.y * h]) - np.array([pinky_tip.x * w, pinky_tip.y * h]))
    ]
    thumb_index_distance = np.linalg.norm(np.array([thumb_tip.x * w, thumb_tip.y * h]) - np.array([index_tip.x * w, index_tip.y * h]))

    # Set thresholds for the distances to determine if the fingers are close to each other and the thumb is away
    close_threshold = 60
    thumb_distance_threshold = 100

    # Check if the hand is open wide and the fingers (except thumb) are close to each other
    if all(dist < close_threshold for dist in tip_distances) and thumb_index_distance > thumb_distance_threshold:
        center_x = int((index_tip.x + middle_tip.x + ring_tip.x) * w / 3)
        center_y = int((index_tip.y + middle_tip.y + ring_tip.y) * h / 3)

        rect_width = 120
        rect_height = 50

        top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), -1)

        return True

    return False
