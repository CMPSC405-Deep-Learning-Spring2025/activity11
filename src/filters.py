import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load the image and check if it exists
image_path = 'cat.jpeg'  # Make sure this path is correct
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check the file path.")

# Define filters
filter_a = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

filter_b = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

filter_c = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Apply any chosen filter
filtered_image = convolve2d(image, filter_a, mode='valid')

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title("Filtered Image")
plt.imshow(filtered_image, cmap='gray')
plt.show()
