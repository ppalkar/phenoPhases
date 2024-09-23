import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler

num_clusters = 20    # Adjust the number of clusters as needed
image = cv2.imread("image.jpg")    # Read the original image

class CustomKernelKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, gamma=0.1, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.random_state = random_state
        self.cluster_centers_ = None

    def _rbf_kernel(self, x, y):
        sq_dist = np.sum(x ** 2) + np.sum(y ** 2) - 2 * np.dot(x, y)
        return np.exp(-self.gamma * sq_dist)

    def plot(self, labels, height, width, image_rgb):
        segmented_image = labels.reshape((height, width))
        plt.figure(figsize=(30, 20))
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.imshow(segmented_image, cmap='tab10')
        plt.title("Segmented Image")
        plt.show()

    def fit_predict(self, X, height, width, image_rgb):
        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        labels = rng.randint(self.n_clusters, size=n_samples)

        for iteration in range(self.max_iter):
            # Update cluster centers based on the current labels
            cluster_centers = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    cluster_centers[k] = np.mean(X[mask], axis=0)

            # Assign points to the nearest cluster center
            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                max_sim = -np.inf
                for k in range(self.n_clusters):
                    sim = self._rbf_kernel(X[i], cluster_centers[k])
                    if sim > max_sim:
                        max_sim = sim
                        new_labels[i] = k

            # print(f"Iteration {iteration}: {np.sum(new_labels)}")      # print objective (maximize)

            # self.plot(new_labels, height, width, image_rgb)            # Plot the segmented image for this iteration

            # Check for convergence
            if np.abs(np.sum(new_labels) - np.sum(labels)) <= self.tol:
                break

            labels = new_labels

        self.cluster_centers_ = cluster_centers
        return labels


# Function to compute LBP
def compute_lbp(image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp


# Convert the image to RGB color space
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

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform Custom Kernel K-means clustering
kernel_kmeans = CustomKernelKMeans(n_clusters=num_clusters, random_state=42, gamma=0.1)
segmented_image = kernel_kmeans.fit_predict(features_scaled, height, width, image_rgb).reshape((height, width))


# Display the final segmented image
plt.figure(figsize=(30, 20))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.subplot(1, 3, 2)
plt.imshow(segmented_image, cmap='tab10')
plt.title("Final Segmented Image")
plt.show()
