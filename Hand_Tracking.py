import cv2
import time
import numpy as np
from tkinter import Text, DISABLED, NORMAL, RIGHT, Y
from calibration import perform_calibration
from drawing_utils import draw_labeled_bounding_boxes, draw_trapezoid, draw_finger_rectangle
from geometry_utils import calculate_distance, calculate_rotation_angle, rotate_bounding_box
from gesture_detection import detect_hand_gesture, detect_ok_gesture, calculate_circle_radius, is_fist_gesture
from smoothing import Smoothing
from hand_model import mp_hands, hands
from boundingbox_utils import draw_bounding_box, remove_bounding_box_for_stroke
from button_utils import draw_start_button, draw_nav_button, is_button_hover, is_button_pinch, draw_gesture_control_button, draw_undo_button, draw_clear_button
from utils import save_to_word_doc

# ML Imports
import torch
from MNIST_digit_recognition import DigitCNN
from custom_train_recognition import CustomDigitCNN
from letter_recognition import LetterCNN
from CROHME_Math_recognition import MathSymbolCNN

def check_collision(line, rect):
    (x1, y1), (x2, y2) = line  # Unpack start and end points from the line tuple
    rect_left, rect_top, rect_right, rect_bottom = rect

    # Simple bounding box collision detection
    if (min(x1, x2) < rect_right and max(x1, x2) > rect_left and
        min(y1, y2) < rect_bottom and max(y1, y2) > rect_top):
        return True
    return False

def run_hand_tracking_gesture_control_1(cap, canvas_strokes, canvas_overlay, root):
    # Button settings
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nav_button_center = (frame_width // 10, frame_height // 2)  # Position of the unlock button for left hand
    nav_button_size = (200, 50)  # Width and height of the nav button
    nav_button_hovered = False
    nav_button_pressed = False
    nav_button_released = True  # Track if the pinch is released

    undo_button_center = (frame_width // 10, frame_height // 2 + 70)  # Below the unlock button
    undo_button_hovered = False
    undo_button_pressed = False
    undo_button_released = True

    calibration_complete = False
    initial_box_area = 1
    right_hand_circle_radius = 10
    locked_circle_radius = 0  # Store the locked circle size
    locked_circle_position = None  # Store the locked circle position
    orientation = ""
    last_adjust_time = time.time()
    last_radius = 0
    adjusting_circle = True
    adjust_timeout = 2.0
    gesture_state = "none"
    circle_locked = False
    last_position = None  # To store the last position of the finger for drawing
    distance_threshold = 7  # Defines the threshold for recognizing proximity of the hand to the screen
    required_consistent_frames = 2  # Defines the number of consistent frames required to recognize a gesture
    leaning_counter = 0  # Counter to track the number of consistent frames
    strokes = []  # List to store all strokes, each stroke is a list of lines
    current_stroke = []  # List to store lines in the current stroke

    # Smoothing for the index finger tip position
    smoothing = Smoothing(method="gaussian", window_size=7, sigma=2)
    # Smoothing for the distance calculation
    distance_smoothing = Smoothing(method="moving_average", window_size=7)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Clear the overlay canvas for each frame
        canvas_overlay.delete("all")

        # Reset button states at the start of each frame
        nav_button_hovered = False
        undo_button_hovered = False

        if not calibration_complete:
            if results.multi_hand_landmarks is None or len(results.multi_hand_landmarks) < 2:
                cv2.putText(frame, "Both hands required for calibration.", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Show OK gesture with left hand to calibrate.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                h, w, _ = frame.shape
                landmark_array = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark])
                x_min, y_min = np.min(landmark_array, axis=0).astype(int)
                x_max, y_max = np.max(landmark_array, axis=0).astype(int)
                current_area = (x_max - x_min) * (y_max - y_min)

                handedness = results.multi_handedness[idx]

                if handedness.classification[0].label == 'Right' and calibration_complete:
                    rotation_angle = calculate_rotation_angle(landmark_array)
                    gesture = detect_hand_gesture(hand_landmarks.landmark, w, h)
                    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                    size = (x_max - x_min, y_max - y_min)
                    rotated_box = rotate_bounding_box(center, size, rotation_angle)
                    cv2.drawContours(frame, [rotated_box], 0, (255, 0, 0), 2)

                    joints = {
                        "4": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                        "3": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                        "2": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                        "1": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    }

                    distance = calculate_distance(current_area, initial_box_area)
                    distance_smoothed = distance_smoothing.apply_smoothing(distance)
                    cv2.putText(frame, f"Distance: {distance_smoothed:.2f} units", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    total_overlap_area = draw_labeled_bounding_boxes(frame, joints, w, h, distance_smoothed, calibration_complete)

                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                    index_palm_width = np.linalg.norm(np.array([index_mcp.x * w, index_mcp.y * h]) - np.array([wrist.x * w, wrist.y * h]))
                    pinky_palm_width = np.linalg.norm(np.array([pinky_mcp.x * w, pinky_mcp.y * h]) - np.array([wrist.x * w, wrist.y * h]))

                    top_left = (int(index_mcp.x * w), int(index_mcp.y * h))
                    top_right = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))
                    bottom_left = (int(wrist.x * w - index_palm_width * 0.5), int(wrist.y * h))
                    bottom_right = (int(wrist.x * w + pinky_palm_width * 0.5), int(wrist.y * h))

                    draw_trapezoid(frame, top_left, top_right, bottom_left, bottom_right)

                    top_width = np.linalg.norm(np.array(top_right) - np.array(top_left))
                    bottom_width = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))

                    rectangle_drawn = draw_finger_rectangle(frame, hand_landmarks.landmark, w, h, calibration_complete)

                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    tip_x = int(index_finger_tip.x * w)
                    tip_y = int(index_finger_tip.y * h)

                    # Apply smoothing to the detected point
                    smoothed_point = smoothing.apply_smoothing((tip_x, tip_y))
                    tip_x, tip_y = int(smoothed_point[0]), int(smoothed_point[1])

                    # Dynamically update line thickness based on the circle radius
                    line_thickness = max(2, int(right_hand_circle_radius / 2))

                    # Check if the index finger is straight up (indicating "away from screen")
                    is_finger_straight_up = detect_hand_gesture(hand_landmarks.landmark, w, h) == "open"

                    # Always show the index finger position on the overlay canvas
                    canvas_overlay.create_oval(
                        tip_x - 5, tip_y - 5,
                        tip_x + 5, tip_y + 5,
                        fill="blue"
                    )

                    if (top_width > bottom_width or total_overlap_area > 100) and distance_smoothed <= distance_threshold and not is_finger_straight_up:
                        # Only show the circle when leaning toward the screen and not straight up
                        leaning_counter += 1
                        if leaning_counter >= required_consistent_frames:
                            orientation = f"Leaning towards the screen ({gesture})"

                            # Draw the circle on the overlay canvas
                            canvas_overlay.create_oval(
                                tip_x - right_hand_circle_radius, tip_y - right_hand_circle_radius,
                                tip_x + right_hand_circle_radius, tip_y + right_hand_circle_radius,
                                outline="green", width=2
                            )

                            # Draw the circle on the camera frame
                            cv2.circle(frame, (tip_x, tip_y), right_hand_circle_radius, (0, 255, 0), 2)

                            if circle_locked:
                                # Only allow drawing when the circle is locked
                                if last_position:
                                    # Draw overlapping circles to create a smooth stroke
                                    steps = int(np.linalg.norm(np.array(last_position) - np.array((tip_x, tip_y))) / (line_thickness / 2))
                                    for i in range(steps):
                                        interp_x = int(last_position[0] + (tip_x - last_position[0]) * (i / steps))
                                        interp_y = int(last_position[1] + (tip_y - last_position[1]) * (i / steps))
                                        line_id = canvas_strokes.create_oval(
                                            interp_x - line_thickness, interp_y - line_thickness,
                                            interp_x + line_thickness, interp_y + line_thickness,
                                            outline="black", width=1, fill="black"
                                        )
                                        current_stroke.append(((last_position[0], last_position[1]), (interp_x, interp_y), line_id))
                                last_position = (tip_x, tip_y)
                            else:
                                # If the circle is unlocked, do not draw on the canvas
                                if current_stroke:
                                    strokes.append(current_stroke)  # Save the completed stroke
                                    current_stroke = []
                                last_position = None  # Reset the last position if not drawing

                    else:
                        leaning_counter = 0
                        orientation = f"Away from the screen ({gesture})"
                        if current_stroke:
                            strokes.append(current_stroke)  # Save the completed stroke
                            current_stroke = []
                        last_position = None  # Reset the last position if not drawing

                    # Show the orientation on the frame
                    cv2.putText(frame, orientation, (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Use rectangle as eraser
                    if rectangle_drawn:
                        # Define the rectangle bounds for collision detection
                        rect_bounds = (
                            int(top_left[0]),
                            int(top_left[1]),
                            int(bottom_right[0]),
                            int(bottom_right[1])
                        )

                        # Check each line for collision with the rectangle
                        strokes_to_erase = []
                        for stroke in strokes:
                            stroke_collides = False
                            lines_to_remove = []
                            for line in stroke:
                                start_point, end_point, line_id = line
                                if check_collision((start_point, end_point), rect_bounds):
                                    canvas_strokes.delete(line_id)  # Delete the line from the canvas
                                    stroke_collides = True
                                    lines_to_remove.append(line)
                            if stroke_collides:
                                for line in lines_to_remove:
                                    stroke.remove(line)
                                if not stroke:
                                    strokes_to_erase.append(stroke)

                        # Remove the completely erased strokes
                        for stroke in strokes_to_erase:
                            strokes.remove(stroke)

                        # Draw the eraser rectangle on the overlay canvas
                        canvas_overlay.create_rectangle(
                            top_left[0], top_left[1], bottom_right[0], bottom_right[1],
                            outline="red", width=2
                        )

                elif handedness.classification[0].label == 'Left':
                    # Allow circle size adjustments with the left hand
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                    if detect_ok_gesture(hand_landmarks.landmark, w, h) and not calibration_complete:
                        initial_top_width, initial_bottom_width, initial_box_area = perform_calibration(hand_landmarks, w, h)
                        calibration_complete = True
                        print("Calibration complete.")

                    if calibration_complete and orientation.startswith("Leaning"):
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        if adjusting_circle:
                            cv2.line(frame, (int(thumb_tip.x * w), int(thumb_tip.y * h)),
                                     (int(index_tip.x * w), int(index_tip.y * h)), (255, 255, 255), 2)

                        current_radius = calculate_circle_radius(hand_landmarks.landmark, w, h)
                        if adjusting_circle:
                            right_hand_circle_radius = current_radius
                            if abs(current_radius - last_radius) < 2:
                                if time.time() - last_adjust_time > adjust_timeout:
                                    adjusting_circle = False
                                    circle_locked = True
                                    locked_circle_radius = right_hand_circle_radius
                                    locked_circle_position = (tip_x, tip_y)
                                    print("Circle adjustment locked.")
                            else:
                                last_adjust_time = time.time()
                            last_radius = current_radius

                        if detect_hand_gesture(hand_landmarks.landmark, w, h) == "open":
                            gesture_state = "open"

                        if gesture_state == "open" and detect_hand_gesture(hand_landmarks.landmark, w, h) == "fist":
                            adjusting_circle = True
                            last_adjust_time = time.time()
                            gesture_state = "none"
                            print("Circle adjustment unlocked.")

        # Show the buttons and handle their interactions
        if calibration_complete:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check if the hand is hovering over the nav button
                    if is_button_hover(hand_landmarks.landmark, frame_width, frame_height, nav_button_center, nav_button_size):
                        nav_button_hovered = True

                    # Check if the pinch gesture is happening over the nav button
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    if is_button_pinch((thumb_tip.x * frame_width, thumb_tip.y * frame_height),
                                       (index_tip.x * frame_width, index_tip.y * frame_height),
                                       nav_button_center, nav_button_size):
                        nav_button_pressed = True
                    else:
                        if nav_button_pressed and nav_button_released:
                            # Unlock the circle size adjustment after the pinch is released
                            adjusting_circle = True
                            circle_locked = False
                            nav_button_pressed = False
                            nav_button_released = False

                    if not nav_button_pressed:
                        nav_button_released = True

                    # Check if hovering and pressing the undo button
                    if is_button_hover(hand_landmarks.landmark, frame_width, frame_height, undo_button_center, nav_button_size):
                        undo_button_hovered = True
                    if is_button_pinch((thumb_tip.x * frame_width, thumb_tip.y * frame_height),
                                       (index_tip.x * frame_width, index_tip.y * frame_height),
                                       undo_button_center, nav_button_size):
                        undo_button_pressed = True
                    else:
                        if undo_button_pressed and undo_button_released:
                            # Perform undo action
                            if strokes:
                                last_stroke = strokes.pop()  # Remove the last stroke from the strokes list
                                for _, _, line_id in last_stroke:
                                    canvas_strokes.delete(line_id)  # Delete each line in the stroke from the canvas
                            undo_button_pressed = False
                            undo_button_released = False

                    if not undo_button_pressed:
                        undo_button_released = True

            # Draw the buttons with the appropriate state
            if calibration_complete and results.multi_hand_landmarks:
                draw_nav_button(frame, nav_button_center, nav_button_size, nav_button_hovered, nav_button_pressed, "UNLOCK")
                draw_undo_button(frame, undo_button_center, nav_button_size, undo_button_hovered, undo_button_pressed)

        # Update the circle status text
        if calibration_complete and results.multi_hand_landmarks:
            circle_status = "Locked" if circle_locked else "Unlocked"
            cv2.putText(frame, f"Circle Status: {circle_status}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()  # Close the Tkinter window when the OpenCV window is closed

def run_hand_tracking_gesture_control_2(cap, canvas_strokes, canvas_overlay, root):
    # Load your pre-trained model
    # model = DigitCNN()
    # model = CustomDigitCNN()
    model = LetterCNN()
    # model = MathSymbolCNN(num_classes=83)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load('digit_recognition_model.pth'))
    # model.load_state_dict(torch.load('custom_digit_recognition_model.pth'))
    model.load_state_dict(torch.load('custom_letter_recognition_model.pth'))
    # model.load_state_dict(torch.load('math_symbol_recognition_model.pth'))
    model.to(device)
    model.eval()  # Set the model to evaluation model

    # Create a text edit screen after gesture control 2 is selected
    text_screen = Text(root, width=20, height=40)
    text_screen.pack(side=RIGHT, fill=Y)
    text_screen.config(state=DISABLED)  # Initially set to read-only

    # Button settings
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nav_button_center = (frame_width // 10, frame_height // 2)  # Position of the unlock button for left hand
    undo_button_center = (frame_width // 10, frame_height // 2 + 70)  # Below the unlock button
    clear_button_center = (frame_width // 10, frame_height // 2 + 140)  # Below the undo button
    button_size = (200, 50)  # Width and height of the buttons

    nav_button_hovered = False
    nav_button_pressed = False
    nav_button_released = True  # Track if the pinch is released

    undo_button_hovered = False
    undo_button_pressed = False
    undo_button_released = True

    clear_button_hovered = False
    clear_button_pressed = False
    clear_button_released = True

    right_hand_circle_radius = 5
    max_circle_radius = 15  # Maximum allowed circle size
    locked_circle_radius = 0  # Store the locked circle size
    locked_circle_position = None  # Store the locked circle position
    orientation = ""
    last_adjust_time = time.time()
    last_radius = 0
    adjusting_circle = True
    adjust_timeout = 3.0
    gesture_state = "none"
    circle_locked = False
    last_position = None  # To store the last position of the finger for drawing

    strokes = []  # List to store all strokes, each stroke is a list of lines
    current_stroke = []  # List to store lines in the current stroke
    bounding_boxes = []  # List to store bounding box information
    label_ids = []  # List to store label order
    label_list = []  # List to store the label texts
    label_counter = 1  # To track the label for bounding boxes
    recognized_digits = []  # List to store recognized digits in order

    # Smoothing for the index finger tip position
    smoothing = Smoothing(method="gaussian", window_size=7, sigma=2)

    calibration_complete = True
    circle_visible = False  # New variable to track circle visibility
    eraser_active = False  # Track if the eraser is active
    drawing_allowed = True  # Track if drawing is allowed (after erasing)

    erase_reset_delay = 1.0  # Delay time after erasing before drawing is allowed again
    last_erase_time = 0  # Track the last time eraser was active

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Clear the overlay canvas for each frame
        canvas_overlay.delete("all")

        # Reset button states at the start of each frame
        nav_button_hovered = False
        undo_button_hovered = False
        clear_button_hovered = False

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                h, w, _ = frame.shape
                handedness = results.multi_handedness[idx]
                hand_label = handedness.classification[0].label

                if hand_label == 'Right':
                    # Right hand: Control circle and drawing/erasing
                    eraser_active = is_fist_gesture(hand_landmarks.landmark)
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    tip_x = int(index_tip.x * w)
                    tip_y = int(index_tip.y * h)

                    # Apply smoothing to the fingertip position
                    smoothed_point = smoothing.apply_smoothing((tip_x, tip_y))
                    tip_x, tip_y = int(smoothed_point[0]), int(smoothed_point[1])

                    # Draw the circle based on the distance between index and middle finger
                    distance_between_fingers = np.linalg.norm(np.array([index_tip.x * w, index_tip.y * h]) - np.array([middle_tip.x * w, middle_tip.y * h]))

                    # Always show the index finger position on the overlay canvas
                    canvas_overlay.create_oval(
                        tip_x - 5, tip_y - 5,
                        tip_x + 5, tip_y + 5,
                        fill="blue"
                    )

                    if not eraser_active and drawing_allowed and distance_between_fingers < 35:  # Only allow drawing when eraser is not active
                        # Show the circle when fingers are close
                        cv2.circle(frame, (tip_x, tip_y), right_hand_circle_radius, (0, 255, 0), 2)  # Draw circle
                        circle_visible = True  # Set circle visibility to True

                        # Draw the circle on the overlay canvas
                        canvas_overlay.create_oval(
                            tip_x - right_hand_circle_radius, tip_y - right_hand_circle_radius,
                            tip_x + right_hand_circle_radius, tip_y + right_hand_circle_radius,
                            outline="green", width=2
                        )

                        if circle_locked:
                            # Dynamically update line thickness based on the circle radius
                            line_thickness = max(2, int(right_hand_circle_radius / 2))

                            if last_position:
                                # Draw overlapping circles to create a smooth stroke
                                steps = int(np.linalg.norm(np.array(last_position) - np.array((tip_x, tip_y))) / (line_thickness / 2))
                                for i in range(steps):
                                    interp_x = int(last_position[0] + (tip_x - last_position[0]) * (i / steps))
                                    interp_y = int(last_position[1] + (tip_y - last_position[1]) * (i / steps))
                                    line_id = canvas_strokes.create_oval(
                                        interp_x - line_thickness, interp_y - line_thickness,
                                        interp_x + line_thickness, interp_y + line_thickness,
                                        outline="black", width=1, fill="black"
                                    )
                                    current_stroke.append((interp_x, interp_y, line_id))
                                last_position = (tip_x, tip_y)
                            else:
                                last_position = (tip_x, tip_y)
                        else:
                            if current_stroke:
                                strokes.append(current_stroke)  # Save the completed stroke
                                draw_bounding_box(canvas_strokes, current_stroke, bounding_boxes, label_counter, label_ids, label_list, text_screen, model, device)  # Draw bounding box
                                label_counter += 1  # Increment the label counter
                                current_stroke = []
                            last_position = None  # Reset last position when not drawing

                    else:
                        circle_visible = False  # Set circle visibility to False
                        if current_stroke:
                            strokes.append(current_stroke)  # Save the completed stroke
                            draw_bounding_box(canvas_strokes, current_stroke, bounding_boxes, label_counter, label_ids, label_list, text_screen, model, device)  # Draw bounding box
                            # class_names = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C', 'Delta', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 'ascii_124', 'b', 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'gamma', 'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta', 'times', 'u', 'v', 'w', 'y', 'z', '{', '}']

                            # Use class_names in the function call
                            # draw_bounding_box(canvas_strokes, current_stroke, bounding_boxes, label_counter, label_ids, label_list, text_screen, model, device, class_names)

                            label_counter += 1  # Increment the label counter
                            current_stroke = []
                        last_position = None  # Reset last position when fingers are not close

                    if eraser_active:
                        # Eraser functionality when the gesture is detected
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                        index_palm_width = np.linalg.norm(np.array([index_mcp.x * w, index_mcp.y * h]) - np.array([wrist.x * w, wrist.y * h]))
                        pinky_palm_width = np.linalg.norm(np.array([pinky_mcp.x * w, pinky_mcp.y * h]) - np.array([wrist.x * w, wrist.y * h]))

                        top_left = (int(index_mcp.x * w), int(index_mcp.y * h))
                        top_right = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))
                        bottom_left = (int(wrist.x * w - index_palm_width * 0.5), int(wrist.y * h))
                        bottom_right = (int(wrist.x * w + pinky_palm_width * 0.5), int(wrist.y * h))

                        # Define the rectangle bounds for collision detection
                        rect_bounds = (
                            min(top_left[0], bottom_left[0]),
                            min(top_left[1], top_right[1]),
                            max(top_right[0], bottom_right[0]),
                            max(bottom_left[1], bottom_right[1])
                        )

                        # Check each stroke for collision with the rectangle
                        strokes_to_erase = []
                        for stroke in strokes:
                            stroke_collides = False
                            lines_to_remove = []
                            for line in stroke:
                                if (rect_bounds[0] <= line[0] <= rect_bounds[2] and
                                    rect_bounds[1] <= line[1] <= rect_bounds[3]):
                                    canvas_strokes.delete(line[2])  # Delete the line from the canvas
                                    stroke_collides = True
                                    lines_to_remove.append(line)
                            if stroke_collides:
                                for line in lines_to_remove:
                                    stroke.remove(line)
                                if not stroke:
                                    strokes_to_erase.append(stroke)

                        # Remove the completely erased strokes
                        for stroke in strokes_to_erase:
                            strokes.remove(stroke)
                            remove_bounding_box_for_stroke(canvas_strokes, stroke, bounding_boxes, label_ids, label_list, model, device, text_screen) 

                        # Draw the eraser rectangle on the overlay canvas
                        canvas_overlay.create_rectangle(
                            top_left[0], top_left[1], bottom_right[0], bottom_right[1],
                            outline="red", width=2
                        )

                        # Disable drawing temporarily after erasing
                        drawing_allowed = False
                        last_erase_time = time.time()

                elif hand_label == 'Left':
                    # Left hand: Handle circle size adjustment, no rectangle or dot should be shown
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    tip_x = int(index_tip.x * w)
                    tip_y = int(index_tip.y * h)

                    if circle_visible:  # Only allow adjustments when the circle is visible
                        if adjusting_circle:
                            cv2.line(frame, (int(thumb_tip.x * w), int(thumb_tip.y * h)),
                                     (int(index_tip.x * w), int(index_tip.y * h)), (255, 255, 255), 2)

                        current_radius = calculate_circle_radius(hand_landmarks.landmark, w, h)
                        if adjusting_circle and not circle_locked:
                            right_hand_circle_radius = min(current_radius, max_circle_radius)  # Limit the circle size to max_circle_radius
                            if abs(current_radius - last_radius) < 2:
                                if time.time() - last_adjust_time > adjust_timeout:
                                    adjusting_circle = False
                                    circle_locked = True
                                    locked_circle_radius = right_hand_circle_radius
                                    locked_circle_position = (tip_x, tip_y)
                                    print("Circle adjustment locked.")
                            else:
                                last_adjust_time = time.time()
                            last_radius = current_radius

                        if detect_hand_gesture(hand_landmarks.landmark, w, h) == "open":
                            gesture_state = "open"

                        if gesture_state == "open" and detect_hand_gesture(hand_landmarks.landmark, w, h) == "fist":
                            adjusting_circle = True
                            last_adjust_time = time.time()
                            gesture_state = "none"
                            print("Circle adjustment unlocked.")

        # Ensure a delay before drawing is allowed again after erasing
        if not drawing_allowed and time.time() - last_erase_time > erase_reset_delay:
            drawing_allowed = True

        # Show the unlock button and handle its interaction
        if calibration_complete:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check if the hand is hovering over the nav button
                    if is_button_hover(hand_landmarks.landmark, frame_width, frame_height, nav_button_center, button_size):
                        nav_button_hovered = True

                    # Check if the pinch gesture is happening over the nav button
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    if is_button_pinch((thumb_tip.x * w, thumb_tip.y * h), (index_tip.x * w, index_tip.y * h), nav_button_center, button_size):
                        nav_button_pressed = True
                    else:
                        if nav_button_pressed and nav_button_released:
                            # Unlock the circle size adjustment after the pinch is released
                            adjusting_circle = True
                            circle_locked = False
                            nav_button_pressed = False
                            nav_button_released = False

                    if not nav_button_pressed:
                        nav_button_released = True

                    # Check if hovering and pressing the undo button
                    if is_button_hover(hand_landmarks.landmark, frame_width, frame_height, undo_button_center, button_size):
                        undo_button_hovered = True
                    if is_button_pinch((thumb_tip.x * w, thumb_tip.y * h), (index_tip.x * w, index_tip.y * h), undo_button_center, button_size):
                        undo_button_pressed = True
                    else:
                        if undo_button_pressed and undo_button_released:
                            # Perform undo action
                            if strokes:
                                last_stroke = strokes.pop()  # Remove the last stroke from the strokes list
                                for line in last_stroke:
                                    canvas_strokes.delete(line[2])  # Delete each line in the stroke from the canvas
                                remove_bounding_box_for_stroke(canvas_strokes, last_stroke, bounding_boxes, label_ids, label_list, model, device, text_screen)  # Remove bounding box
                            undo_button_pressed = False
                            undo_button_released = False

                    if not undo_button_pressed:
                        undo_button_released = True

                    if is_button_hover(hand_landmarks.landmark, frame_width, frame_height, clear_button_center, button_size):
                        clear_button_hovered = True

                    if is_button_pinch((thumb_tip.x * w, thumb_tip.y * h),
                                    (index_tip.x * w, index_tip.y * h),
                                    clear_button_center, button_size):
                        clear_button_pressed = True
                    else:
                        if clear_button_pressed and clear_button_released:
                            # Temporarily enable the text screen to clear it
                            text_screen.config(state=NORMAL)
                            
                            # Get the content from the text screen
                            text_content = text_screen.get('1.0', 'end-1c')
                            
                            # Save the content to the Word document only if there's content
                            if text_content.strip():
                                save_to_word_doc(text_content)
                            
                            # Clear the canvas and associated stroke data
                            canvas_strokes.delete("all")
                            strokes.clear()
                            bounding_boxes.clear()
                            label_ids.clear()
                            label_list.clear()
                            recognized_digits.clear()
                            
                            # Clear the text screen and disable it again
                            text_screen.delete('1.0', 'end')
                            text_screen.config(state=DISABLED)
                            
                            print("Canvas and text screen cleared, content saved to the document if it was present")
                            clear_button_pressed = False
                            clear_button_released = False

                    if not clear_button_pressed:
                        clear_button_released = True

            # Draw the buttons
            if calibration_complete and results.multi_hand_landmarks:
                draw_nav_button(frame, nav_button_center, button_size, nav_button_hovered, nav_button_pressed, "UNLOCK")
                draw_undo_button(frame, undo_button_center, button_size, undo_button_hovered, undo_button_pressed)
                draw_clear_button(frame, clear_button_center, button_size, clear_button_hovered, clear_button_pressed)

        # Update the circle status text
        if calibration_complete and results.multi_hand_landmarks:
            circle_status = "Locked" if circle_locked else "Unlocked"
            cv2.putText(frame, f"Circle Status: {circle_status}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()  # Close the Tkinter window when the OpenCV window is closed

def run_hand_tracking(cap, canvas_strokes, canvas_overlay, root, model):
    # Button settings
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    button_center = (frame_width // 2, frame_height // 2)  # Position of the start button
    button_size = (200, 100)  # Width and height of the start button
    button_hovered = False
    button_pressed = False
    calibration_started = False

    gesture_control_buttons = [
        {"text": "Gesture control 1", "center": (frame_width // 3, frame_height // 2), "size": (300, 150)},
        {"text": "Gesture control 2", "center": (2 * frame_width // 3, frame_height // 2), "size": (300, 150)},
    ]
    gesture_control_selected = None  # To track which control is selected
    gesture_control_released = True  # To ensure pinch is released before taking action

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Clear the overlay canvas for each frame
        canvas_overlay.delete("all")

        # Reset hover status at the start of each frame
        button_hovered = False

        if not calibration_started:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape

                    # Check if the hand is hovering over the button
                    if is_button_hover(hand_landmarks.landmark, w, h, button_center, button_size):
                        button_hovered = True

                    # Check if the pinch gesture is happening
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    if is_button_pinch((thumb_tip.x * w, thumb_tip.y * h), (index_tip.x * w, index_tip.y * h), button_center, button_size):
                        button_pressed = True
                    else:
                        if button_pressed:
                            # Calibration starts after the pinch is released
                            calibration_started = True
                            button_pressed = False

            # Draw the start button with the appropriate state
            draw_start_button(frame, button_center, button_size, button_hovered, button_pressed)

        elif gesture_control_selected is None:
            # Show gesture control buttons if calibration has started but no gesture control has been selected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for button in gesture_control_buttons:
                        hovered = is_button_hover(hand_landmarks.landmark, frame_width, frame_height, button["center"], button["size"])
                        pressed = False

                        # Check if the pinch gesture is happening over the button
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        if is_button_pinch((thumb_tip.x * frame_width, thumb_tip.y * frame_height),
                                           (index_tip.x * frame_width, index_tip.y * frame_height),
                                           button["center"], button["size"]):
                            pressed = True

                        if pressed and gesture_control_released:
                            gesture_control_selected = button["text"]
                            gesture_control_released = False

                            if gesture_control_selected == "Gesture control 1":
                                run_hand_tracking_gesture_control_1(cap, canvas_strokes, canvas_overlay, root)
                                return  # Exit this function after running gesture control 1

                            elif gesture_control_selected == "Gesture control 2":
                                run_hand_tracking_gesture_control_2(cap, canvas_strokes, canvas_overlay, root)
                                return  # Exit this function after running gesture control 2

                        if not pressed:
                            gesture_control_released = True

                        draw_gesture_control_button(frame, button["center"], button["size"], hovered, pressed, button["text"])

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()  # Close the Tkinter window when the OpenCV window is closed