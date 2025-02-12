import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).parent.absolute()
image = cv2.imread(project_path/'images'/'like_lenna.png', cv2.IMREAD_GRAYSCALE)

# Create Prewitt Kernel for X
kx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Create Prewitt Kernel for Y
ky = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

# Use prewitt kernel
gx = cv2.filter2D(image, -1, kx)
gy = cv2.filter2D(image, -1, ky)

# Result image
prewitt_result = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

# Display result
plt.imshow(prewitt_result, cmap='gray')
plt.axis('off')
plt.title('Prewitt Edge Detection')
plt.show()
