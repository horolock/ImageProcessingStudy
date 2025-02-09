import cv2
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).parent.absolute()
image = cv2.imread(project_path/'images'/'like_lenna.png', cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def gaussian_pyramid(image, levels):
    pyramid = [image]

    for i in range(levels - 1):
        image =cv2.pyrDown(image)
        pyramid.append(image)
    
    return pyramid 

levels = 5
pyramid = gaussian_pyramid(image_rgb, levels)

# Visualize gaussian pyramid
fig, axes = plt.subplots(1, levels, figsize=(20, 8))

for i, ax in enumerate(axes):
    ax.imshow(pyramid[i])
    ax.axis('off')
    ax.set_title(f'Level {i + 1}')

plt.tight_layout()
plt.show()