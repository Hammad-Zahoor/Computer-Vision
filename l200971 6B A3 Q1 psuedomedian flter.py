# All the problem andits cases are handled
# Psuedomedian filter

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def calculate_M(L):
    return (L + 1) // 2

def MAXMIN(sequence, M):
    chunked_sequence = [sequence[i:i+M] for i in range(0, len(sequence) - M + 1)]
    min_values = [min(chunk) for chunk in chunked_sequence]
    return max(min_values)

def MINMAX(sequence, M):
    chunked_sequence = [sequence[i:i+M] for i in range(0, len(sequence) - M + 1)]
    max_values = [max(chunk) for chunk in chunked_sequence]
    return min(max_values)

def pseudo_median_filter(image, filter_size=3):
    height, width = image.shape
    pad = filter_size // 2
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            neighborhood = image[i - pad:i + pad + 1, j - pad:j + pad + 1].flatten()
            sorted_neighborhood = np.sort(neighborhood)
            L = len(sorted_neighborhood)
            M = calculate_M(L)
            max_min_value = MAXMIN(sorted_neighborhood, M)
            min_max_value = MINMAX(sorted_neighborhood, M)
            pseudo_median = 0.5 * max_min_value + 0.5 * min_max_value
            result[i, j] = pseudo_median

    return result

def process_image():
    filename = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp *.gif')])
    if filename:
        input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        window_size = int(window_entry.get())
        filtered_image = pseudo_median_filter(input_image, filter_size=window_size)

        # Resize images if needed to fit in the window
        input_image = cv2.resize(input_image, (500, 500))
        filtered_image = cv2.resize(filtered_image, (500, 500))

        cv2.imshow('Original Image', input_image)
        cv2.imshow('Filtered Image', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Create GUI window
root = tk.Tk()
root.title('Pseudo Median Filter')
root.geometry('500x500')

# Browse button
browse_button = tk.Button(root, text='Browse Image', command=process_image)
browse_button.pack()

# Window size entry
window_label = tk.Label(root, text='Enter Window Size:')
window_label.pack()
window_entry = tk.Entry(root)
window_entry.pack()

root.mainloop()
