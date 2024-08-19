import mediapipe as mp

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,          # Use False for real-time video processing
    max_num_hands=2,                  # Maximum number of hands to detect
    min_detection_confidence=0.8,     # Higher confidence threshold for hand detection
    min_tracking_confidence=0.5,      # Increase tracking confidence for more consistent hand tracking
    model_complexity=1                # Use the higher complexity model for better accuracy
)