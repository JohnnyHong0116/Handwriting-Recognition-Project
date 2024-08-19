import numpy as np
from hand_model import mp_hands

def perform_calibration(landmarks, w, h):
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    index_palm_width = np.linalg.norm(np.array([index_mcp.x * w, index_mcp.y * h]) - np.array([wrist.x * w, wrist.y * h]))
    pinky_palm_width = np.linalg.norm(np.array([pinky_mcp.x * w, pinky_mcp.y * h]) - np.array([wrist.x * w, wrist.y * h]))

    initial_top_width = np.linalg.norm(np.array([index_mcp.x * w, index_mcp.y * h]) - np.array([pinky_mcp.x * w, pinky_mcp.y * h]))
    initial_bottom_width = np.linalg.norm(np.array([wrist.x * w - index_palm_width * 0.5, wrist.y * h]) - np.array([wrist.x * w + pinky_palm_width * 0.5, wrist.y * h]))
    initial_box_area = (index_palm_width + pinky_palm_width) * (index_mcp.y * h - wrist.y * h)

    return initial_top_width, initial_bottom_width, initial_box_area
