
# DIGIT RECOGNITION

import os
import glob
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageEnhance
import torch
from torchvision import transforms
from tkinter import DISABLED, NORMAL

saved_files = {}  # Dictionary to track the filenames of saved images by label
recognized_digits = []  # List to store the recognized digits

def clear_unused_images(output_dir, label_list):
    files = glob.glob(os.path.join(output_dir, 'bounding_box_*.png'))
    for file in files:
        file_label = os.path.basename(file).split('_')[-1].split('.')[0]
        if file_label not in label_list:
            try:
                os.remove(file)
                saved_files.pop(file_label, None)
                print(f"Removed unused image: {file}")
            except OSError as e:
                print(f"Error removing {file}: {e.strerror}")

def recognize_digit_with_model(image, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(img)
        predicted_digit = output.argmax(dim=1).item()  # Get the predicted digit
    return str(predicted_digit)

def update_text_screen(text_screen, recognized_digits):
    text_screen.config(state=NORMAL)
    text_screen.delete('1.0', 'end')  # Clear the current content
    text_screen.insert('end', ''.join(recognized_digits))  # Insert the recognized digits
    # Make the text screen read-only
    text_screen.config(state=DISABLED)

def save_stroke_image(canvas_strokes, x1, y1, x2, y2, label):
    padding = 5
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2 += padding
    y2 += padding

    canvas_strokes.update_idletasks()
    canvas_width = canvas_strokes.winfo_width()
    canvas_height = canvas_strokes.winfo_height()
    img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(img)

    for item in canvas_strokes.find_all():
        coords = canvas_strokes.coords(item)
        if len(coords) == 4:
            fill = canvas_strokes.itemcget(item, "fill")
            if fill:
                fill_rgb = ImageColor.getrgb(fill)
                line_width = int(float(canvas_strokes.itemcget(item, "width")))
                draw_circular_stroke(draw, coords, line_width, fill_rgb)

    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH_MORE)
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH)
    cropped_img = cropped_img.filter(ImageFilter.BLUR)
    enhancer = ImageEnhance.Contrast(cropped_img)
    cropped_img = enhancer.enhance(2.5)
    cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
    cropped_img = cropped_img.convert("L")
    threshold = 128
    cropped_img = cropped_img.point(lambda p: p > threshold and 255)
    cropped_img = cropped_img.convert("RGB")

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"bounding_box_{label}.png")
    cropped_img.save(filename)
    saved_files[label] = filename
    print(f"Saved stroke image as {filename}")

def update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen):
    label_list.clear()
    recognized_digits.clear()  # Clear the recognized digits list

    for i, (_, _, label_id) in enumerate(bounding_boxes):
        new_label = str(i + 1)
        canvas_strokes.itemconfig(label_id, text=new_label)
        label_list.append(new_label)

    print("Updated label list:", label_list)

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    for _, bbox_id, label_id in bounding_boxes:
        label = canvas_strokes.itemcget(label_id, 'text')
        x1, y1, x2, y2 = canvas_strokes.coords(bbox_id)
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label)

        image_path = saved_files[label]
        img = Image.open(image_path)
        recognized_digit = recognize_digit_with_model(img, model, device)
        recognized_digits.append(recognized_digit)

    update_text_screen(text_screen, recognized_digits)  # Update the text edit screen

    clear_unused_images(output_dir, label_list)
    print("Recognized digits:", recognized_digits)

def merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen, distance_threshold=20):
    merged_boxes = []
    used_indices = set()

    for i, (group1, bbox_id1, label_id1) in enumerate(bounding_boxes):
        if i in used_indices:
            continue

        x11, y11, x12, y12 = canvas_strokes.coords(bbox_id1)
        merged_group = group1
        merged_bbox_id = bbox_id1
        merged_label_id = label_id1

        for j, (group2, bbox_id2, label_id2) in enumerate(bounding_boxes):
            if i != j and j not in used_indices:
                x21, y21, x22, y22 = canvas_strokes.coords(bbox_id2)

                if not (x12 < x21 - distance_threshold or x22 < x11 - distance_threshold or
                        y12 < y21 - distance_threshold or y22 < y11 - distance_threshold):
                    new_x1, new_y1 = min(x11, x21), min(y11, y21)
                    new_x2, new_y2 = max(x12, x22), max(y12, y22)

                    merged_group += group2
                    canvas_strokes.delete(merged_bbox_id)
                    canvas_strokes.delete(bbox_id2)
                    merged_bbox_id = canvas_strokes.create_rectangle(new_x1, new_y1, new_x2, new_y2, outline="red")

                    if int(canvas_strokes.itemcget(label_id1, "text")) < int(canvas_strokes.itemcget(label_id2, "text")):
                        canvas_strokes.delete(label_id2)
                        final_label = canvas_strokes.itemcget(label_id1, 'text')
                    else:
                        canvas_strokes.delete(label_id1)
                        final_label = canvas_strokes.itemcget(label_id2, 'text')
                        merged_label_id = label_id2

                    used_indices.add(j)

        used_indices.add(i)
        merged_boxes.append((merged_group, merged_bbox_id, merged_label_id))

        if len(merged_group) > len(group1):
            new_top_left = (new_x1, new_y1)
            new_top_right = (new_x2, new_y1)
            new_bottom_left = (new_x1, new_y2)
            new_bottom_right = (new_x2, new_y2)
            print(f"Updated Label {final_label}:")
            print(f"  Merged Coordinates: Top-Left {new_top_left}, Top-Right {new_top_right}, Bottom-Left {new_bottom_left}, Bottom-Right {new_bottom_right}")

            save_stroke_image(canvas_strokes, new_x1, new_y1, new_x2, new_y2, final_label)

    bounding_boxes.clear()
    bounding_boxes.extend(merged_boxes)

    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen)

def draw_bounding_box(canvas_strokes, stroke_group, bounding_boxes, label_counter, label_ids, label_list, text_screen, model, device, save_images=True):
    x_coords = []
    y_coords = []

    for line in stroke_group:
        if isinstance(line, tuple) and len(line) == 3:
            x, y, _ = line
            x_coords.append(x)
            y_coords.append(y)

    if not x_coords or not y_coords:
        return

    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    bbox_id = canvas_strokes.create_rectangle(x1, y1, x2, y2, outline="red")
    label_id = canvas_strokes.create_text((x1 + x2) // 2, y1 - 10, text=str(label_counter), fill="blue")

    bounding_boxes.append((stroke_group, bbox_id, label_id))
    label_ids.append(label_id)
    label_list.append(str(label_counter))

    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)

    print(f"Label {label_counter}:")
    print(f"  Coordinates: Top-Left {top_left}, Top-Right {top_right}, Bottom-Left {bottom_left}, Bottom-Right {bottom_right}")
    print("  Current label list:", label_list)

    if save_images:
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label_counter)

    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen)

    return top_left, top_right, bottom_left, bottom_right

def draw_circular_stroke(draw, coords, width, fill):
    for i in range(0, len(coords) - 2, 2):
        draw.ellipse(
            [coords[i] - width // 2, coords[i + 1] - width // 2,
             coords[i] + width // 2, coords[i + 1] + width // 2],
            fill=fill
        )
        if i + 2 < len(coords):
            draw.line(
                [coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                fill=fill, width=width
            )
    if len(coords) % 2 == 0:
        draw.ellipse(
            [coords[-2] - width // 2, coords[-1] - width // 2,
             coords[-2] + width // 2, coords[-1] + width // 2],
            fill=fill
        )

def remove_bounding_box_for_stroke(canvas_strokes, stroke_group, bounding_boxes, label_ids, label_list, model, device, text_screen):
    to_remove = []

    for i, (stroke, bbox_id, label_id) in enumerate(bounding_boxes):
        if set(stroke_group).issubset(set(stroke)):
            stroke[:] = [s for s in stroke if s not in stroke_group]
            if not stroke:
                canvas_strokes.delete(bbox_id)
                canvas_strokes.delete(label_id)
                to_remove.append(i)

    for index in sorted(to_remove, reverse=True):
        bounding_boxes.pop(index)
        label_ids.pop(index)
        label_list.pop(index)
        recognized_digits.pop(index)  # Remove the corresponding recognized digit

    print("Updated label list after removal:", label_list)

    update_text_screen(text_screen, recognized_digits)  # Update the text edit screen
    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen)
    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen)


'''
# LETTER RECOGNITION
import os
import glob
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageEnhance
import torch
from torchvision import transforms
from tkinter import DISABLED, NORMAL

saved_files = {}  # Dictionary to track the filenames of saved images by label
recognized_letters = []  # List to store the recognized letters

def clear_unused_images(output_dir, label_list):
    files = glob.glob(os.path.join(output_dir, 'bounding_box_*.png'))
    for file in files:
        file_label = os.path.basename(file).split('_')[-1].split('.')[0]
        if file_label not in label_list:
            try:
                os.remove(file)
                saved_files.pop(file_label, None)
                print(f"Removed unused image: {file}")
            except OSError as e:
                print(f"Error removing {file}: {e.strerror}")

def recognize_letter_with_model(image, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(img)
        predicted_index = output.argmax(dim=1).item()  # Get the predicted index
        letter = chr(predicted_index + ord('A'))  # Convert the index to a letter (A-Z)
    return letter

def update_text_screen(text_screen, recognized_letters):
     # Clear the current content
    text_screen.config(state=NORMAL)
    text_screen.delete('1.0', 'end')  # Clear the current content
    text_screen.insert('end', ''.join(recognized_letters))  # Insert the recognized letters
    # Make the text screen read-only
    text_screen.config(state=DISABLED)

def save_stroke_image(canvas_strokes, x1, y1, x2, y2, label):
    padding = 5
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2 += padding
    y2 += padding

    canvas_strokes.update_idletasks()
    canvas_width = canvas_strokes.winfo_width()
    canvas_height = canvas_strokes.winfo_height()
    img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(img)

    for item in canvas_strokes.find_all():
        coords = canvas_strokes.coords(item)
        if len(coords) == 4:
            fill = canvas_strokes.itemcget(item, "fill")
            if fill:
                fill_rgb = ImageColor.getrgb(fill)
                line_width = int(float(canvas_strokes.itemcget(item, "width")))
                draw_circular_stroke(draw, coords, line_width, fill_rgb)

    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH_MORE)
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH)
    cropped_img = cropped_img.filter(ImageFilter.BLUR)
    enhancer = ImageEnhance.Contrast(cropped_img)
    cropped_img = enhancer.enhance(2.5)
    cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
    cropped_img = cropped_img.convert("L")
    threshold = 128
    cropped_img = cropped_img.point(lambda p: p > threshold and 255)
    cropped_img = cropped_img.convert("RGB")

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"bounding_box_{label}.png")
    cropped_img.save(filename)
    saved_files[label] = filename
    print(f"Saved stroke image as {filename}")

def update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen):
    label_list.clear()
    recognized_letters.clear()  # Clear the recognized letters list

    for i, (_, _, label_id) in enumerate(bounding_boxes):
        new_label = str(i + 1)
        canvas_strokes.itemconfig(label_id, text=new_label)
        label_list.append(new_label)

    print("Updated label list:", label_list)

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    for _, bbox_id, label_id in bounding_boxes:
        label = canvas_strokes.itemcget(label_id, 'text')
        x1, y1, x2, y2 = canvas_strokes.coords(bbox_id)
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label)

        image_path = saved_files[label]
        img = Image.open(image_path)
        recognized_letter = recognize_letter_with_model(img, model, device)
        recognized_letters.append(recognized_letter)

    update_text_screen(text_screen, recognized_letters)  # Update the text edit screen

    clear_unused_images(output_dir, label_list)
    print("Recognized letters:", recognized_letters)

def merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen, distance_threshold=20):
    merged_boxes = []
    used_indices = set()

    for i, (group1, bbox_id1, label_id1) in enumerate(bounding_boxes):
        if i in used_indices:
            continue

        x11, y11, x12, y12 = canvas_strokes.coords(bbox_id1)
        merged_group = group1
        merged_bbox_id = bbox_id1
        merged_label_id = label_id1

        for j, (group2, bbox_id2, label_id2) in enumerate(bounding_boxes):
            if i != j and j not in used_indices:
                x21, y21, x22, y22 = canvas_strokes.coords(bbox_id2)

                if not (x12 < x21 - distance_threshold or x22 < x11 - distance_threshold or
                        y12 < y21 - distance_threshold or y22 < y11 - distance_threshold):
                    new_x1, new_y1 = min(x11, x21), min(y11, y21)
                    new_x2, new_y2 = max(x12, x22), max(y12, y22)

                    merged_group += group2
                    canvas_strokes.delete(merged_bbox_id)
                    canvas_strokes.delete(bbox_id2)
                    merged_bbox_id = canvas_strokes.create_rectangle(new_x1, new_y1, new_x2, new_y2, outline="red")

                    if int(canvas_strokes.itemcget(label_id1, "text")) < int(canvas_strokes.itemcget(label_id2, "text")):
                        canvas_strokes.delete(label_id2)
                        final_label = canvas_strokes.itemcget(label_id1, 'text')
                    else:
                        canvas_strokes.delete(label_id1)
                        final_label = canvas_strokes.itemcget(label_id2, 'text')
                        merged_label_id = label_id2

                    used_indices.add(j)

        used_indices.add(i)
        merged_boxes.append((merged_group, merged_bbox_id, merged_label_id))

        if len(merged_group) > len(group1):
            new_top_left = (new_x1, new_y1)
            new_top_right = (new_x2, new_y1)
            new_bottom_left = (new_x1, new_y2)
            new_bottom_right = (new_x2, new_y2)
            print(f"Updated Label {final_label}:")
            print(f"  Merged Coordinates: Top-Left {new_top_left}, Top-Right {new_top_right}, Bottom-Left {new_bottom_left}, Bottom-Right {new_bottom_right}")

            save_stroke_image(canvas_strokes, new_x1, new_y1, new_x2, new_y2, final_label)

    bounding_boxes.clear()
    bounding_boxes.extend(merged_boxes)

    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen)

def draw_bounding_box(canvas_strokes, stroke_group, bounding_boxes, label_counter, label_ids, label_list, text_screen, model, device, save_images=True):
    x_coords = []
    y_coords = []

    for line in stroke_group:
        if isinstance(line, tuple) and len(line) == 3:
            x, y, _ = line
            x_coords.append(x)
            y_coords.append(y)

    if not x_coords or not y_coords:
        return

    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    bbox_id = canvas_strokes.create_rectangle(x1, y1, x2, y2, outline="red")
    label_id = canvas_strokes.create_text((x1 + x2) // 2, y1 - 10, text=str(label_counter), fill="blue")

    bounding_boxes.append((stroke_group, bbox_id, label_id))
    label_ids.append(label_id)
    label_list.append(str(label_counter))

    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)

    print(f"Label {label_counter}:")
    print(f"  Coordinates: Top-Left {top_left}, Top-Right {top_right}, Bottom-Left {bottom_left}, Bottom-Right {bottom_right}")
    print("  Current label list:", label_list)

    if save_images:
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label_counter)

    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen)

    return top_left, top_right, bottom_left, bottom_right

def draw_circular_stroke(draw, coords, width, fill):
    for i in range(0, len(coords) - 2, 2):
        draw.ellipse(
            [coords[i] - width // 2, coords[i + 1] - width // 2,
             coords[i] + width // 2, coords[i + 1] + width // 2],
            fill=fill
        )
        if i + 2 < len(coords):
            draw.line(
                [coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                fill=fill, width=width
            )
    if len(coords) % 2 == 0:
        draw.ellipse(
            [coords[-2] - width // 2, coords[-1] - width // 2,
             coords[-2] + width // 2, coords[-1] + width // 2],
            fill=fill
        )

def remove_bounding_box_for_stroke(canvas_strokes, stroke_group, bounding_boxes, label_ids, label_list, model, device, text_screen):
    to_remove = []

    for i, (stroke, bbox_id, label_id) in enumerate(bounding_boxes):
        if set(stroke_group).issubset(set(stroke)):
            stroke[:] = [s for s in stroke if s not in stroke_group]
            if not stroke:
                canvas_strokes.delete(bbox_id)
                canvas_strokes.delete(label_id)
                to_remove.append(i)

    for index in sorted(to_remove, reverse=True):
        bounding_boxes.pop(index)
        label_ids.pop(index)
        label_list.pop(index)
        recognized_letters.pop(index)  # Remove the corresponding recognized letter

    print("Updated label list after removal:", label_list)

    update_text_screen(text_screen, recognized_letters)  # Update the text edit screen
    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen)
    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen)
'''

'''
import os
import glob
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageEnhance
import torch
from torchvision import transforms
from tkinter import DISABLED, NORMAL

saved_files = {}  # Dictionary to track the filenames of saved images by label
recognized_symbols = []  # List to store the recognized symbols

def clear_unused_images(output_dir, label_list):
    files = glob.glob(os.path.join(output_dir, 'bounding_box_*.png'))
    for file in files:
        file_label = os.path.basename(file).split('_')[-1].split('.')[0]
        if file_label not in label_list:
            try:
                os.remove(file)
                saved_files.pop(file_label, None)
                print(f"Removed unused image: {file}")
            except OSError as e:
                print(f"Error removing {file}: {e.strerror}")

def recognize_symbol_with_model(image, model, device, class_names):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(img)
        predicted_index = output.argmax(dim=1).item()  # Get the predicted index
        symbol = class_names[predicted_index]  # Convert the index to the corresponding symbol
    return symbol

def update_text_screen(text_screen, recognized_symbols):
    # Clear the current content
    text_screen.config(state=NORMAL)
    text_screen.delete('1.0', 'end')  # Clear the current content
    text_screen.insert('end', ''.join(recognized_symbols))  # Insert the recognized symbols
    # Make the text screen read-only
    text_screen.config(state=DISABLED)

def save_stroke_image(canvas_strokes, x1, y1, x2, y2, label):
    padding = 5
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2 += padding
    y2 += padding

    canvas_strokes.update_idletasks()
    canvas_width = canvas_strokes.winfo_width()
    canvas_height = canvas_strokes.winfo_height()
    img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(img)

    for item in canvas_strokes.find_all():
        coords = canvas_strokes.coords(item)
        if len(coords) == 4:
            fill = canvas_strokes.itemcget(item, "fill")
            if fill:
                fill_rgb = ImageColor.getrgb(fill)
                line_width = int(float(canvas_strokes.itemcget(item, "width")))
                draw_circular_stroke(draw, coords, line_width, fill_rgb)

    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH_MORE)
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH)
    cropped_img = cropped_img.filter(ImageFilter.BLUR)
    enhancer = ImageEnhance.Contrast(cropped_img)
    cropped_img = enhancer.enhance(2.5)
    cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
    cropped_img = cropped_img.convert("L")
    threshold = 128
    cropped_img = cropped_img.point(lambda p: p > threshold and 255)
    cropped_img = cropped_img.convert("RGB")

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"bounding_box_{label}.png")
    cropped_img.save(filename)
    saved_files[label] = filename
    print(f"Saved stroke image as {filename}")

def update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen, class_names):
    label_list.clear()
    recognized_symbols.clear()  # Clear the recognized symbols list

    for i, (_, _, label_id) in enumerate(bounding_boxes):
        new_label = str(i + 1)
        canvas_strokes.itemconfig(label_id, text=new_label)
        label_list.append(new_label)

    print("Updated label list:", label_list)

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    for _, bbox_id, label_id in bounding_boxes:
        label = canvas_strokes.itemcget(label_id, 'text')
        x1, y1, x2, y2 = canvas_strokes.coords(bbox_id)
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label)

        image_path = saved_files[label]
        img = Image.open(image_path)
        recognized_symbol = recognize_symbol_with_model(img, model, device, class_names)
        recognized_symbols.append(recognized_symbol)

    update_text_screen(text_screen, recognized_symbols)  # Update the text edit screen

    clear_unused_images(output_dir, label_list)
    print("Recognized symbols:", recognized_symbols)

def merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen, class_names, distance_threshold=20):
    merged_boxes = []
    used_indices = set()

    for i, (group1, bbox_id1, label_id1) in enumerate(bounding_boxes):
        if i in used_indices:
            continue

        x11, y11, x12, y12 = canvas_strokes.coords(bbox_id1)
        merged_group = group1
        merged_bbox_id = bbox_id1
        merged_label_id = label_id1

        for j, (group2, bbox_id2, label_id2) in enumerate(bounding_boxes):
            if i != j and j not in used_indices:
                x21, y21, x22, y22 = canvas_strokes.coords(bbox_id2)

                if not (x12 < x21 - distance_threshold or x22 < x11 - distance_threshold or
                        y12 < y21 - distance_threshold or y22 < y11 - distance_threshold):
                    new_x1, new_y1 = min(x11, x21), min(y11, y21)
                    new_x2, new_y2 = max(x12, x22), max(y12, y22)

                    merged_group += group2
                    canvas_strokes.delete(merged_bbox_id)
                    canvas_strokes.delete(bbox_id2)
                    merged_bbox_id = canvas_strokes.create_rectangle(new_x1, new_y1, new_x2, new_y2, outline="red")

                    if int(canvas_strokes.itemcget(label_id1, "text")) < int(canvas_strokes.itemcget(label_id2, "text")):
                        canvas_strokes.delete(label_id2)
                        final_label = canvas_strokes.itemcget(label_id1, 'text')
                    else:
                        canvas_strokes.delete(label_id1)
                        final_label = canvas_strokes.itemcget(label_id2, 'text')
                        merged_label_id = label_id2

                    used_indices.add(j)

        used_indices.add(i)
        merged_boxes.append((merged_group, merged_bbox_id, merged_label_id))

        if len(merged_group) > len(group1):
            new_top_left = (new_x1, new_y1)
            new_top_right = (new_x2, new_y1)
            new_bottom_left = (new_x1, new_y2)
            new_bottom_right = (new_x2, new_y2)
            print(f"Updated Label {final_label}:")
            print(f"  Merged Coordinates: Top-Left {new_top_left}, Top-Right {new_top_right}, Bottom-Left {new_bottom_left}, Bottom-Right {new_bottom_right}")

            save_stroke_image(canvas_strokes, new_x1, new_y1, new_x2, new_y2, final_label)

    bounding_boxes.clear()
    bounding_boxes.extend(merged_boxes)

    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen, class_names)

def draw_bounding_box(canvas_strokes, stroke_group, bounding_boxes, label_counter, label_ids, label_list, text_screen, model, device, class_names, save_images=True):
    x_coords = []
    y_coords = []

    for line in stroke_group:
        if isinstance(line, tuple) and len(line) == 3:
            x, y, _ = line
            x_coords.append(x)
            y_coords.append(y)

    if not x_coords or not y_coords:
        return

    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    bbox_id = canvas_strokes.create_rectangle(x1, y1, x2, y2, outline="red")
    label_id = canvas_strokes.create_text((x1 + x2) // 2, y1 - 10, text=str(label_counter), fill="blue")

    bounding_boxes.append((stroke_group, bbox_id, label_id))
    label_ids.append(label_id)
    label_list.append(str(label_counter))

    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)

    print(f"Label {label_counter}:")
    print(f"  Coordinates: Top-Left {top_left}, Top-Right {top_right}, Bottom-Left {bottom_left}, Bottom-Right {bottom_right}")
    print("  Current label list:", label_list)

    if save_images:
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label_counter)

    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen, class_names)

    return top_left, top_right, bottom_left, bottom_right

def draw_circular_stroke(draw, coords, width, fill):
    for i in range(0, len(coords) - 2, 2):
        draw.ellipse(
            [coords[i] - width // 2, coords[i + 1] - width // 2,
             coords[i] + width // 2, coords[i + 1] + width // 2],
            fill=fill
        )
        if i + 2 < len(coords):
            draw.line(
                [coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                fill=fill, width=width
            )
    if len(coords) % 2 == 0:
        draw.ellipse(
            [coords[-2] - width // 2, coords[-1] - width // 2,
             coords[-2] + width // 2, coords[-1] + width // 2],
            fill=fill
        )

def remove_bounding_box_for_stroke(canvas_strokes, stroke_group, bounding_boxes, label_ids, label_list, model, device, text_screen, class_names):
    to_remove = []

    for i, (stroke, bbox_id, label_id) in enumerate(bounding_boxes):
        if set(stroke_group).issubset(set(stroke)):
            stroke[:] = [s for s in stroke if s not in stroke_group]
            if not stroke:
                canvas_strokes.delete(bbox_id)
                canvas_strokes.delete(label_id)
                to_remove.append(i)

    for index in sorted(to_remove, reverse=True):
        bounding_boxes.pop(index)
        label_ids.pop(index)
        label_list.pop(index)
        recognized_symbols.pop(index)  # Remove the corresponding recognized symbol

    print("Updated label list after removal:", label_list)

    update_text_screen(text_screen, recognized_symbols)  # Update the text edit screen
    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen, class_names)
    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen, class_names)
'''

'''
import os
import glob
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageEnhance
import torch
from torchvision import transforms
from tkinter import DISABLED, NORMAL

saved_files = {}
recognized_characters = []

def clear_unused_images(output_dir, label_list):
    files = glob.glob(os.path.join(output_dir, 'bounding_box_*.png'))
    for file in files:
        file_label = os.path.basename(file).split('_')[-1].split('.')[0]
        if file_label not in label_list:
            try:
                os.remove(file)
                saved_files.pop(file_label, None)
                print(f"Removed unused image: {file}")
            except OSError as e:
                print(f"Error removing {file}: {e.strerror}")

def recognize_character_with_model(image, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(img)
        predicted_index = output.argmax(dim=1).item()  # Get the predicted index

    # If the predicted index is 0-9, return digit; if 10-35, return letter
    if 0 <= predicted_index <= 9:
        return str(predicted_index)  # Digits 0-9
    else:
        return chr(predicted_index - 10 + ord('A'))  # Letters A-Z


def update_text_screen(text_screen, recognized_characters):
    text_screen.config(state=NORMAL)
    text_screen.delete('1.0', 'end')
    text_screen.insert('end', ''.join(recognized_characters))
    text_screen.config(state=DISABLED)

def save_stroke_image(canvas_strokes, x1, y1, x2, y2, label):
    padding = 5
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2 += padding
    y2 += padding

    canvas_strokes.update_idletasks()
    canvas_width = canvas_strokes.winfo_width()
    canvas_height = canvas_strokes.winfo_height()
    img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(img)

    for item in canvas_strokes.find_all():
        coords = canvas_strokes.coords(item)
        if len(coords) == 4:
            fill = canvas_strokes.itemcget(item, "fill")
            if fill:
                fill_rgb = ImageColor.getrgb(fill)
                line_width = int(float(canvas_strokes.itemcget(item, "width")))
                draw_circular_stroke(draw, coords, line_width, fill_rgb)

    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH_MORE)
    cropped_img = cropped_img.filter(ImageFilter.SMOOTH)
    cropped_img = cropped_img.filter(ImageFilter.BLUR)
    enhancer = ImageEnhance.Contrast(cropped_img)
    cropped_img = enhancer.enhance(2.5)
    cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
    cropped_img = cropped_img.convert("L")
    threshold = 128
    cropped_img = cropped_img.point(lambda p: p > threshold and 255)
    cropped_img = cropped_img.convert("RGB")

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"bounding_box_{label}.png")
    cropped_img.save(filename)
    saved_files[label] = filename
    print(f"Saved stroke image as {filename}")

def update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen):
    label_list.clear()
    recognized_characters.clear()

    for i, (_, _, label_id) in enumerate(bounding_boxes):
        new_label = str(i + 1)
        canvas_strokes.itemconfig(label_id, text=new_label)
        label_list.append(new_label)

    print("Updated label list:", label_list)

    output_dir = os.path.join(os.getcwd(), "saved_strokes")
    for _, bbox_id, label_id in bounding_boxes:
        label = canvas_strokes.itemcget(label_id, 'text')
        x1, y1, x2, y2 = canvas_strokes.coords(bbox_id)
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label)

        image_path = saved_files[label]
        img = Image.open(image_path)
        recognized_character = recognize_character_with_model(img, model, device)
        recognized_characters.append(recognized_character)

    update_text_screen(text_screen, recognized_characters)
    clear_unused_images(output_dir, label_list)
    print("Recognized characters:", recognized_characters)

def merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen, distance_threshold=20):
    merged_boxes = []
    used_indices = set()

    for i, (group1, bbox_id1, label_id1) in enumerate(bounding_boxes):
        if i in used_indices:
            continue

        x11, y11, x12, y12 = canvas_strokes.coords(bbox_id1)
        merged_group = group1
        merged_bbox_id = bbox_id1
        merged_label_id = label_id1

        for j, (group2, bbox_id2, label_id2) in enumerate(bounding_boxes):
            if i != j and j not in used_indices:
                x21, y21, x22, y22 = canvas_strokes.coords(bbox_id2)

                if not (x12 < x21 - distance_threshold or x22 < x11 - distance_threshold or
                        y12 < y21 - distance_threshold or y22 < y11 - distance_threshold):
                    new_x1, new_y1 = min(x11, x21), min(y11, y21)
                    new_x2, new_y2 = max(x12, x22), max(y12, y22)

                    merged_group += group2
                    canvas_strokes.delete(merged_bbox_id)
                    canvas_strokes.delete(bbox_id2)
                    merged_bbox_id = canvas_strokes.create_rectangle(new_x1, new_y1, new_x2, new_y2, outline="red")

                    if int(canvas_strokes.itemcget(label_id1, "text")) < int(canvas_strokes.itemcget(label_id2, "text")):
                        canvas_strokes.delete(label_id2)
                        final_label = canvas_strokes.itemcget(label_id1, 'text')
                    else:
                        canvas_strokes.delete(label_id1)
                        final_label = canvas_strokes.itemcget(label_id2, 'text')
                        merged_label_id = label_id2

                    used_indices.add(j)

        used_indices.add(i)
        merged_boxes.append((merged_group, merged_bbox_id, merged_label_id))

        if len(merged_group) > len(group1):
            new_top_left = (new_x1, new_y1)
            new_top_right = (new_x2, new_y1)
            new_bottom_left = (new_x1, new_y2)
            new_bottom_right = (new_x2, new_y2)
            print(f"Updated Label {final_label}:")
            print(f"  Merged Coordinates: Top-Left {new_top_left}, Top-Right {new_top_right}, Bottom-Left {new_bottom_left}, Bottom-Right {new_bottom_right}")

            save_stroke_image(canvas_strokes, new_x1, new_y1, new_x2, new_y2, final_label)

    bounding_boxes.clear()
    bounding_boxes.extend(merged_boxes)

    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen)

def draw_bounding_box(canvas_strokes, stroke_group, bounding_boxes, label_counter, label_ids, label_list, text_screen, model, device, save_images=True):
    x_coords = []
    y_coords = []

    for line in stroke_group:
        if isinstance(line, tuple) and len(line) == 3:
            x, y, _ = line
            x_coords.append(x)
            y_coords.append(y)

    if not x_coords or not y_coords:
        return

    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    bbox_id = canvas_strokes.create_rectangle(x1, y1, x2, y2, outline="red")
    label_id = canvas_strokes.create_text((x1 + x2) // 2, y1 - 10, text=str(label_counter), fill="blue")

    bounding_boxes.append((stroke_group, bbox_id, label_id))
    label_ids.append(label_id)
    label_list.append(str(label_counter))

    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)

    print(f"Label {label_counter}:")
    print(f"  Coordinates: Top-Left {top_left}, Top-Right {top_right}, Bottom-Left {bottom_left}, Bottom-Right {bottom_right}")
    print("  Current label list:", label_list)

    if save_images:
        save_stroke_image(canvas_strokes, x1, y1, x2, y2, label_counter)

    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen)

    return top_left, top_right, bottom_left, bottom_right

def draw_circular_stroke(draw, coords, width, fill):
    for i in range(0, len(coords) - 2, 2):
        draw.ellipse(
            [coords[i] - width // 2, coords[i + 1] - width // 2,
             coords[i] + width // 2, coords[i + 1] + width // 2],
            fill=fill
        )
        if i + 2 < len(coords):
            draw.line(
                [coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                fill=fill, width=width
            )
    if len(coords) % 2 == 0:
        draw.ellipse(
            [coords[-2] - width // 2, coords[-1] - width // 2,
             coords[-2] + width // 2, coords[-1] + width // 2],
            fill=fill
        )

def remove_bounding_box_for_stroke(canvas_strokes, stroke_group, bounding_boxes, label_ids, label_list, model, device, text_screen):
    to_remove = []

    for i, (stroke, bbox_id, label_id) in enumerate(bounding_boxes):
        if set(stroke_group).issubset(set(stroke)):
            stroke[:] = [s for s in stroke if s not in stroke_group]
            if not stroke:
                canvas_strokes.delete(bbox_id)
                canvas_strokes.delete(label_id)
                to_remove.append(i)

    for index in sorted(to_remove, reverse=True):
        bounding_boxes.pop(index)
        label_ids.pop(index)
        label_list.pop(index)
        recognized_characters.pop(index)  # Remove the corresponding recognized character

    print("Updated label list after removal:", label_list)

    update_text_screen(text_screen, recognized_characters)  # Update the text edit screen
    update_labels(canvas_strokes, bounding_boxes, label_list, model, device, text_screen)
    merge_bounding_boxes(canvas_strokes, bounding_boxes, label_ids, label_list, model, device, text_screen)
'''
