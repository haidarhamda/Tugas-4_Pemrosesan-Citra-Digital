import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import yolo

selected_processing_function = "yolo"
uploaded_file_path = None
window_width = 1440
window_height = 1000

def upload_image():
    global uploaded_file_path
    temp_uploaded_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if temp_uploaded_file_path:
        uploaded_file_path = temp_uploaded_file_path
        process_image()

def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim)

def process_image():
    global uploaded_file_path
    if not uploaded_file_path:
        return

    original = cv2.imread(uploaded_file_path)
    if original is None:
        return

    resized_original = resize(original, width=350)

    if selected_processing_function == "yolo":
        processed_image = yolo.yolo(original)
        processed_image = resize(processed_image, width=900)

    original_rgb = cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB)
    original_pil = Image.fromarray(original_rgb)

    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_pil = Image.fromarray(processed_rgb)

    original_tk = ImageTk.PhotoImage(original_pil)
    processed_tk = ImageTk.PhotoImage(processed_pil)

    original_label.config(image=original_tk)
    original_label.image = original_tk

    processed_label.config(image=processed_tk)
    processed_label.image = processed_tk

def set_process():
    global selected_processing_function
    selected_processing_function = processing_var.get()
    process_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f'{window_width}x{window_height}')

    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    upload_button.pack(pady=20)

    processing_var = tk.StringVar(value="yolo")
    radio_frame = tk.Frame(root)
    radio_frame.pack(pady=10)

    tk.Radiobutton(radio_frame, text="YOLO", variable=processing_var, value="yolo", command=set_process).grid(row=0, column=1, padx=5)

    original_label = Label(root, text="Original Image")
    original_label.place(x=20, y=50)

    processed_label = Label(root, text="Processed Image")
    processed_label.place(x=window_width//2, y=(window_height+window_height//3)//2, anchor="center")

    root.mainloop()
