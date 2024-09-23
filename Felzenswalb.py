import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage.util import img_as_float

# Read the original image
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to float
image_float = img_as_float(image_rgb)

# Apply Felzenszwalb segmentation
segments = felzenszwalb(image_float, scale=50, sigma=3, min_size=30)

# Create a color-coded segmentation image with different colors for each segment
segmented_image = label2rgb(segments, image_rgb, kind='overlay')

# Display the original and segmented images
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title("Segmented Image with Different Colors")
plt.show()
