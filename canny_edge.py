import cv2
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).parent.absolute()
image = cv2.imread(project_path/'images'/'like_lenna.png', cv2.IMREAD_GRAYSCALE)

# Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
# Canny edge
canny_edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

# Example 1 Visualize
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(image, cmap='gray')
# axes[0].axis('off')
# axes[0].set_title('Original Image')

# axes[1].imshow(blurred_image, cmap='gray')
# axes[1].axis('off')
# axes[1].set_title('Gaussian Blurred Image')

# axes[2].imshow(canny_edges, cmap='gray')
# axes[2].axis('off')
# axes[2].set_title('Canny Edge Detection')


# Example 2 Visualize
thresholds = [(10, 50), (50, 100), (100, 150), (150, 200)]
canny_results = []

for threshold1, threshold2 in thresholds:
    canny_image = cv2.Canny(blurred_image, threshold1=threshold1, threshold2=threshold2)
    canny_results.append(canny_image)

fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 5))

for i, ax in enumerate(axes):
    ax.imshow(canny_results[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'threshold1={thresholds[i][0]}, threshold2={thresholds[i][1]}')

plt.show()