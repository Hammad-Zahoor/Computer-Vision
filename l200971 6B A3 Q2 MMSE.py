import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def compute_local_means(image, window_size):  # computing local mean
    pad = window_size // 2
    height, width = image.shape
    local_means = np.zeros((height, width), dtype=np.float32)

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            window = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            local_means[i, j] = np.mean(window)

    return local_means

def compute_local_variances(image, local_means, window_size):  # computing local variance
    pad = window_size // 2
    height, width = image.shape
    local_variances = np.zeros((height, width), dtype=np.float32)

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            window = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            local_variances[i, j] = np.sum(np.square(window - local_means[i, j])) / (window_size**2 - 1)

    return local_variances

def MMSE_filter(image, window_size=3, noise_variance=0.1):
    height, width = image.shape
    pad = window_size // 2
    result = np.zeros((height, width), dtype=np.uint8)

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Compute local means and variances
    local_means = compute_local_means(padded_image.astype(np.float32), window_size)
    local_variances = compute_local_variances(padded_image.astype(np.float32), local_means, window_size)

    # Apply the MMSE filter
    for i in range(height):
        for j in range(width):
            if local_variances[i+pad, j+pad] < 1e-6:  # Check for near-zero local variance
                result[i, j] = image[i, j]  # Skip filtering for this pixel
            else:
                weight = noise_variance**2 / (local_variances[i+pad, j+pad]**2) # + noise_variance**2)
                result[i, j] = np.clip(np.round(image[i, j] - weight * (image[i, j] - local_means[i+pad, j+pad])), 0, 255).astype(np.uint8)

    return result

def compute_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_mse(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse

def browse_image():
    filename = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp *.gif')])
    if filename:
        input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        window_size = int(window_entry.get())
        noise_variance = float(noise_entry.get())
        filtered_image = MMSE_filter(input_image, window_size, noise_variance)

        # Calculate PSNR and MSE
        psnr_value = compute_psnr(input_image, filtered_image)
        mse_value = compute_mse(input_image, filtered_image)

        # Resize images if needed to fit in the window
        input_image = cv2.resize(input_image, (500, 500))
        filtered_image = cv2.resize(filtered_image, (500, 500))

        cv2.imshow('Original Image', input_image)
        cv2.imshow('Filtered Image', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display PSNR and MSE values
        psnr_label.config(text=f'PSNR: {psnr_value:.2f}')
        mse_label.config(text=f'MSE: {mse_value:.2f}')

# Create GUI window
root = tk.Tk()
root.title('MMSE Filter')
root.geometry('500x600')  # Larger window size

# Browse button
browse_button = tk.Button(root, text='Browse Image', command=browse_image)
browse_button.pack()

# Window size entry
window_label = tk.Label(root, text='Enter Window Size:')
window_label.pack()
window_entry = tk.Entry(root)
window_entry.pack()

# Noise variance entry
noise_label = tk.Label(root, text='Enter Noise Variance:')
noise_label.pack()
noise_entry = tk.Entry(root)
noise_entry.pack()

# PSNR label
psnr_label = tk.Label(root, text='PSNR: ')
psnr_label.pack()

# MSE label
mse_label = tk.Label(root, text='MSE: ')
mse_label.pack()

root.mainloop()
