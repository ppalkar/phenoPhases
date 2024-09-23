import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
from skimage.feature import local_binary_pattern

num_clusters = 20  # Adjust the number of clusters as needed

# Function to compute LBP
def compute_lbp(image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

# Read the original image
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to LAB color space
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# Compute LBP for the grayscale version of the image
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
lbp_image = compute_lbp(gray_image)

# Reshape the images to 2D arrays
pixel_values = image_lab.reshape((-1, 3))
lbp_values = lbp_image.reshape((-1, 1))

# Normalize pixel values to the range [0, 1]
pixel_values = np.float32(pixel_values) / 255.0

# Add spatial information (x, y coordinates) to the feature space
height, width, _ = image.shape
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
spatial_features = np.stack((x_coords.flatten(), y_coords.flatten()), axis=1)

# Concatenate all features
features = np.concatenate((pixel_values, spatial_features, lbp_values), axis=1)

# Perform Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=num_clusters, random_state=42).fit(features)
segmented_image = gmm.predict(features).reshape((height, width))

# Display the original and segmented images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='tab10')
plt.title("Segmented Image")

plt.show()
