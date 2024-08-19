import cv2

def draw_start_button(frame, center, size, hovered, pressed):
    color = (255, 0, 0)  # Blue color by default
    fill_color = (173, 216, 230)  # Lighter blue when hovered
    shadow_color = (0, 0, 139)  # Dark blue when pressed

    top_left = (center[0] - size[0] // 2, center[1] - size[1] // 2)
    bottom_right = (center[0] + size[0] // 2, center[1] + size[1] // 2)

    if hovered:
        cv2.rectangle(frame, top_left, bottom_right, fill_color, -1)  # Fill rectangle when hovered
    if pressed:
        cv2.rectangle(frame, top_left, bottom_right, shadow_color, -1)  # Shadow effect when pressed

    cv2.rectangle(frame, top_left, bottom_right, color, 2)  # Rectangle outline

    text = "START"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness)

def draw_nav_button(frame, center, size, hovered, pressed, text):
    color = (255, 0, 0)  # Blue color by default
    fill_color = (173, 216, 230)  # Lighter blue when hovered
    shadow_color = (0, 0, 139)  # Dark blue when pressed

    top_left = (center[0] - size[0] // 2, center[1] - size[1] // 2)
    bottom_right = (center[0] + size[0] // 2, center[1] + size[1] // 2)

    if hovered:
        cv2.rectangle(frame, top_left, bottom_right, fill_color, -1)  # Fill rectangle when hovered
    if pressed:
        cv2.rectangle(frame, top_left, bottom_right, shadow_color, -1)  # Shadow effect when pressed

    cv2.rectangle(frame, top_left, bottom_right, color, 2)  # Rectangle outline

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness)

def is_button_hover(landmarks, frame_width, frame_height, button_center, button_size):
    for lm in landmarks:
        lm_x = int(lm.x * frame_width)
        lm_y = int(lm.y * frame_height)
        if (button_center[0] - button_size[0] // 2 < lm_x < button_center[0] + button_size[0] // 2 and
            button_center[1] - button_size[1] // 2 < lm_y < button_center[1] + button_size[1] // 2):
            return True
    return False

def is_button_pinch(thumb_tip, index_tip, button_center, button_size):
    pinch_distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
    if (button_center[0] - button_size[0] // 2 < thumb_tip[0] < button_center[0] + button_size[0] // 2 and
        button_center[1] - button_size[1] // 2 < thumb_tip[1] < button_center[1] + button_size[1] // 2):
        return pinch_distance < 30
    return False

def draw_gesture_control_button(frame, center, size, hovered, pressed, text):
    color = (255, 0, 0)  # Blue color by default
    fill_color = (173, 216, 230)  # Lighter blue when hovered
    shadow_color = (0, 0, 139)  # Dark blue when pressed

    top_left = (center[0] - size[0] // 2, center[1] - size[1] // 2)
    bottom_right = (center[0] + size[0] // 2, center[1] + size[1] // 2)

    if hovered:
        cv2.rectangle(frame, top_left, bottom_right, fill_color, -1)  # Fill rectangle when hovered
    if pressed:
        cv2.rectangle(frame, top_left, bottom_right, shadow_color, -1)  # Shadow effect when pressed

    cv2.rectangle(frame, top_left, bottom_right, color, 2)  # Rectangle outline

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness)

def draw_undo_button(frame, center, size, hovered, pressed):
    color = (0, 255, 0)  # Green color by default
    fill_color = (144, 238, 144)  # Lighter green when hovered
    shadow_color = (34, 139, 34)  # Dark green when pressed

    top_left = (center[0] - size[0] // 2, center[1] - size[1] // 2)
    bottom_right = (center[0] + size[0] // 2, center[1] + size[1] // 2)

    if hovered:
        cv2.rectangle(frame, top_left, bottom_right, fill_color, -1)  # Fill rectangle when hovered
    if pressed:
        cv2.rectangle(frame, top_left, bottom_right, shadow_color, -1)  # Shadow effect when pressed

    cv2.rectangle(frame, top_left, bottom_right, color, 2)  # Rectangle outline

    text = "UNDO"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness)


def draw_clear_button(frame, center, size, hovered, pressed):
    color = (0, 255, 0)  # Green color by default
    fill_color = (144, 238, 144)  # Lighter green when hovered
    shadow_color = (34, 139, 34)  # Dark green when pressed

    top_left = (center[0] - size[0] // 2, center[1] - size[1] // 2)
    bottom_right = (center[0] + size[0] // 2, center[1] + size[1] // 2)

    if hovered:
        cv2.rectangle(frame, top_left, bottom_right, fill_color, -1)  # Fill rectangle when hovered
    if pressed:
        cv2.rectangle(frame, top_left, bottom_right, shadow_color, -1)  # Shadow effect when pressed

    text = "CLEAR"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness)
