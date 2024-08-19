import numpy as np
import cv2
from hand_model import mp_hands

def calculate_distance(box_area, initial_area):
    return initial_area / box_area if box_area else 0

def calculate_rotation_angle(landmark_array):
    wrist = landmark_array[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmark_array[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def rotate_bounding_box(center, size, angle):
    rect = ((float(center[0]), float(center[1])), (float(size[0]), float(size[1])), angle)
    box = cv2.boxPoints(rect)
    return np.intp(box)

def calculate_overlap_area(box1, box2):
    overlap_top_left = (max(box1[0][0], box2[0][0]), max(box1[0][1], box2[0][1]))
    overlap_bottom_right = (min(box1[1][0], box2[1][0]), min(box1[1][1], box2[1][1]))

    if overlap_top_left[0] < overlap_bottom_right[0] and overlap_top_left[1] < overlap_bottom_right[1]:
        overlap_width = overlap_bottom_right[0] - overlap_top_left[0]
        overlap_height = overlap_bottom_right[1] - overlap_top_left[1]
        return overlap_width * overlap_height
    return 0
