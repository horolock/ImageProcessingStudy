import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).parent.absolute()
image = cv2.imread(project_path/'images'/'like_lenna.png', cv2.IMREAD_GRAYSCALE)

# Sobel kernel
gx_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gy_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Result image
sobel_result = cv2.addWeighted(gx_sobel, 0.5, gy_sobel, 0.5, 0)

# Display result
plt.imshow(sobel_result, cmap='gray')
plt.axis('off')
plt.title('Sobel Edge Detection')
plt.show()
