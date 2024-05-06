import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Entry, Label
from PIL import Image, ImageTk

def apply_autumn_color_map(gray_image):
    # Define Autumn color map with 256 entries
    rainbow_map = np.zeros((256, 3), dtype=np.uint8)

    for i in range(0, 43):
        rainbow_map[i] = [0, 0, np.clip(int(i * 6), 0, 255)]  # Blue to Cyan
    for i in range(43, 128):
        rainbow_map[i] = [0, np.clip(int((i - 42) * 6), 0, 255), 255]  # Cyan to Yellow
    for i in range(128, 213):
        rainbow_map[i] = [np.clip(int((i - 127) * 6), 0, 255), 255, np.clip(255 - int((i - 127) * 6), 0, 255)]  # Yellow to Orange
    for i in range(213, 256):
        rainbow_map[i] = [255, np.clip(255 - int((i - 212) * 6), 0, 255), 0]  # Orange to Red

    # Normalize the grayscale image to [0, 1] range
    normalized_image = gray_image / 255.0

    # Apply Autumn color map manually
    colorized_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    for i in range(256):
        colorized_image[normalized_image * 255 == i] = rainbow_map[i]

    return colorized_image

def retinex_single_scale(image, sigma):
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    # Calculate the single-scale Retinex response
    response = np.log(image + 1) - np.log(blur + 1)
    return response

def retinex_multi_scale(image, sigma_list, weights):
    # Convert image to float32 for processing
    image = image.astype(np.float32)
    # Initialize RMSR response
    rmsr_response = np.zeros_like(image)
    # Apply Retinex-like Single-Scale Retinex (SSR) at each scale and compute RMSR
    for sigma, weight in zip(sigma_list, weights):
        response = retinex_single_scale(image, sigma)
        rmsr_response += weight * response
    return rmsr_response

def msrcr(image, sigma_list, weights, b):
    # Compute RMSR response using multi-scale Retinex
    rmsr_response = retinex_multi_scale(image, sigma_list, weights)
    # Detail enhancement
    detail = b * rmsr_response
    # Color restoration
    restored = image + detail
    # Clip and convert back to uint8
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    return restored

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Get values from entry widgets
        b_val = float(b_entry.get())
        weights_val = [float(val.strip()) for val in weights_entry.get().split(',')]
        sigma_val = [int(val.strip()) for val in sigma_entry.get().split(',')]

        enhanced_image = msrcr(original_image, sigma_val, weights_val, b_val)
        enhanced_colorized = apply_autumn_color_map(enhanced_image)
        display_images(original_image,enhanced_image, enhanced_colorized)

def display_images(original,enhanced, colorized):
    # Resize images for display
    original_resized = cv2.resize(original, (300, 300))
    enhanced_resized = cv2.resize(enhanced, (300, 300))
    colorized_resized = cv2.resize(colorized, (300, 300))

    # Display images in GUI
    original_img = Image.fromarray(original_resized)
    enahnced_img = Image.fromarray(enhanced_resized)
    colorized_img = Image.fromarray(colorized_resized)

    original_photo = ImageTk.PhotoImage(original_img)
    enhanced_photo = ImageTk.PhotoImage(enahnced_img)
    colorized_photo = ImageTk.PhotoImage(colorized_img)

    original_label.config(image=original_photo)
    original_label.image = original_photo
    enhanced_label.config(image=enhanced_photo)
    enhanced_label.image = enhanced_photo
    colorized_label.config(image=colorized_photo)
    colorized_label.image = colorized_photo

# Initialize GUI
root = tk.Tk()
root.title("Image Enhancement and Pseudo-Coloring")
root.geometry("1400x700")

# Entry widgets for parameters with labels and padding
b_label = Label(root, text="b(25) :")
b_label.pack(pady=10)
b_entry = Entry(root)
b_entry.pack(pady=10)
weights_label = Label(root, text="Weights (comma-separated) (0.3,0.3,0.3) :")
weights_label.pack(pady=10)
weights_entry = Entry(root)
weights_entry.pack(pady=10)
sigma_label = Label(root, text="Sigma List (comma-separated)(25,80,250) :")
sigma_label.pack(pady=10)
sigma_entry = Entry(root)
sigma_entry.pack(pady=10)

# Browse button
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Frames for displaying images
original_frame = tk.Frame(root, width=300, height=300)
original_frame.pack(side=tk.LEFT, padx=20)
enhanced_frame = tk.Frame(root, width=300, height=300)
enhanced_frame.pack(side=tk.LEFT, padx=20)
colorized_frame = tk.Frame(root, width=300, height=300)
colorized_frame.pack(side=tk.RIGHT, padx=20)

# Labels for displaying images
original_label = tk.Label(original_frame)
original_label.pack()
enhanced_label = tk.Label(enhanced_frame)
enhanced_label.pack()
colorized_label = tk.Label(colorized_frame)
colorized_label.pack()

root.mainloop()
