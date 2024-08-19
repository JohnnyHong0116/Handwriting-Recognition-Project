import torch
from tkinter import Tk, Canvas
import cv2
import threading
import win32gui
import win32con
import win32api
from Hand_Tracking import run_hand_tracking
from MNIST_digit_recognition import DigitCNN
from custom_train_recognition import CustomDigitCNN
from letter_recognition import LetterCNN
from CROHME_Math_recognition import MathSymbolCNN

def set_canvas_transparent(canvas, colorkey=(0, 0, 0)):
    hwnd = canvas.winfo_id()
    colorkey = win32api.RGB(*colorkey)
    wnd_exstyle = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    new_exstyle = wnd_exstyle | win32con.WS_EX_LAYERED
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, new_exstyle)
    win32gui.SetLayeredWindowAttributes(hwnd, colorkey, 255, win32con.LWA_COLORKEY)

def main():
    # model = DigitCNN()
    # model = CustomDigitCNN()
    model = LetterCNN()
    # model = MathSymbolCNN(num_classes=83)

    # model.load_state_dict(torch.load('digit_recognition_model.pth'))
    # model.load_state_dict(torch.load('custom_digit_recognition_model.pth'))
    model.load_state_dict(torch.load('custom_letter_recognition_model.pth'))
    # model.load_state_dict(torch.load('math_symbol_recognition_model.pth'))
    model.eval()

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Tkinter setup
    root = Tk()
    root.title("Virtual Whiteboard")
    root.geometry('1280x720')

    canvas_strokes = Canvas(root, width=1280, height=720, bg='white')
    canvas_strokes.place(x=0, y=0)

    canvas_overlay = Canvas(root, width=1280, height=720, bg='#000000', highlightthickness=0)
    canvas_overlay.place(x=0, y=0)
    set_canvas_transparent(canvas_overlay, colorkey=(0, 0, 0))

    tracking_thread = threading.Thread(target=run_hand_tracking, args=(cap, canvas_strokes, canvas_overlay, root, model))
    tracking_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()
